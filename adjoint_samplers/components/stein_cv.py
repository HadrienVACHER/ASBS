import torch
import torch.nn as nn


def mean_free_basis(n_particles, spatial_dim):
    """ Orthonormal basis of configurations with zero center of mass.

    Flat layout matches the rest of the repo: index p * spatial_dim + s.
    The bridge noise is an isotropic Gaussian on this subspace, so Stein's
    identity is the ordinary divergence in these coordinates.
    """
    particle = torch.eye(n_particles, dtype=torch.float32)
    proj = particle - torch.ones(n_particles, n_particles) / n_particles
    evals, evecs = torch.linalg.eigh(proj)
    particle_basis = evecs[:, evals > 0.5]  # (n_particles, n_particles - 1)

    ambient = n_particles * spatial_dim
    rank = (n_particles - 1) * spatial_dim
    basis = torch.zeros(ambient, rank, dtype=torch.float32)
    col = 0
    for s in range(spatial_dim):
        for j in range(n_particles - 1):
            for p in range(n_particles):
                basis[p * spatial_dim + s, col] = particle_basis[p, j]
            col += 1
    return basis


class ConcatMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_layers, zero_last=False):
        super().__init__()
        layers = []
        width = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, *parts):
        return self.net(torch.cat(parts, dim=-1))


def partial_divergence(values, y, create_graph):
    """ (div)_i = ∂ values_i / ∂ y_i. values and y are (B, K). """
    parts = []
    for i in range(y.shape[-1]):
        grad = torch.autograd.grad(
            values[:, i].sum(),
            y,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(y)
        parts.append(grad[:, i])
    return torch.stack(parts, dim=-1)


def time_bins(t, n_bins):
    return (t.detach().squeeze(-1) * n_bins).long().clamp(0, n_bins - 1)


def bin_center(values, t, n_bins):
    """ Subtract the detached per-bin batch mean, binned in t.

    The subtracted constant depends on the bin only. Its expectation over
    batches is E[values | bin], so the controller update stays unbiased in
    expectation, and the test field gains nothing from a constant offset.
    It removes marginal bias only and adds O(1/sqrt(B)) noise.
    """
    idx = time_bins(t, n_bins)
    centered = torch.zeros_like(values)
    for b in range(n_bins):
        mask = (idx == b).unsqueeze(-1)
        count = mask.sum().clamp(min=1.0)
        mean = (values.detach() * mask).sum(dim=0) / count
        centered = centered + mask * (values - mean)
    return centered


def bridge_score_y1(tau, total_var, y0, yt, y1, tau_max):
    """ ∇_{y1} log p_base(yt | y0, y1) on the mean-free coordinates.

    Zero past tau_max, where the bridge is treated as a Dirac and the only
    legal control variate is zero. The active branch uses the unclamped τ.
    """
    active = tau < tau_max
    denom = total_var * (1.0 - tau)
    residual = yt - (1.0 - tau) * y0 - tau * y1
    score = residual / denom.clamp(min=1e-6)
    return torch.where(active, score, torch.zeros_like(score)), active


class SteinControlVariate(nn.Module):
    """ Bridge Stein control variate on the center-of-mass-free subspace.

    Given (t, x0, xt), the remaining randomness is x1. Its conditional score is
    s_br + ∇_{y1} log p(y1 | y0), where s_br is the VE bridge score and the
    second term is the endpoint conditional of the controlled process. That
    second term is not -∇E. A separate network is fit to it by conditional
    Hyvärinen score matching on buffer pairs (x0, x1). The accuracy of that
    fit is the only source of bias in the control variate.

    λ(t, x0, xt) is a function of the conditioning variables, so it cannot
    change the conditional mean. λ and the score network start at zero.
    """

    def __init__(
        self,
        n_particles,
        spatial_dim,
        hidden=64,
        score_hidden=128,
        n_layers=3,
        tau_max=0.999,
        n_bins=4,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.tau_max = tau_max
        self.n_bins = n_bins
        rank = (n_particles - 1) * spatial_dim
        self.rank = rank
        self.register_buffer("Q", mean_free_basis(n_particles, spatial_dim))

        # λ starts at 0 so the first update equals vanilla ASBS. The field must
        # not start at 0: with TF = 0 and λ = 0 neither gets a gradient.
        self.field = ConcatMLP(3 * rank + 2, rank, hidden, n_layers)
        self.score_net = ConcatMLP(2 * rank, rank, score_hidden, n_layers, zero_last=True)
        self.lam_net = ConcatMLP(1 + 2 * rank, 1, hidden, n_layers, zero_last=True)

    def cv_parameters(self):
        return list(self.field.parameters()) + list(self.lam_net.parameters())

    def to_coord(self, x):
        return x @ self.Q

    def to_ambient(self, y):
        return y @ self.Q.T

    def _tau(self, ref_sde, t):
        total_var = ref_sde.total_var
        if not torch.is_tensor(total_var):
            total_var = torch.tensor(total_var, device=t.device, dtype=t.dtype)
        else:
            total_var = total_var.to(device=t.device, dtype=t.dtype)
        tau = ref_sde._diffsquare_integral(t) / total_var
        return tau, total_var

    def stein_field(self, ref_sde, t, x0, xt, x1, energy):
        """ TF in subspace coordinates, with a graph only into the test field.

        s_br and the learned endpoint score are coefficients. t, x0, xt are
        parameters of the conditional law. Divergence is in y1 only.
        """
        tau, total_var = self._tau(ref_sde, t)
        y0 = self.to_coord(x0).detach()
        yt = self.to_coord(xt).detach()
        y1 = self.to_coord(x1).detach().requires_grad_(True)
        t = t.detach()

        s_br, active = bridge_score_y1(
            tau, total_var, y0, yt, y1.detach(), self.tau_max,
        )
        with torch.no_grad():
            endpoint_score = self.score_net(y1.detach(), y0)
        score = s_br + endpoint_score

        x1_var = self.to_ambient(y1)
        energy_value = energy.eval(x1_var)
        if energy_value.ndim == 1:
            energy_value = energy_value.unsqueeze(-1)
        field = self.field(y1, energy_value, t, y0, yt)
        tf = partial_divergence(field, y1, create_graph=True) + field * score

        # (1-τ) is a function of t, so it preserves a conditional mean of zero.
        # It cancels the 1/(1-τ) pole of s_br.
        scale = (1.0 - tau).detach()
        tf = torch.where(active, tf * scale, torch.zeros_like(tf))
        return tf, tau.detach()

    def hyvarinen(self, x0, x1):
        """ Conditional implicit score matching for ∇_{y1} log p(y1 | y0). """
        y0 = self.to_coord(x0).detach()
        y1 = self.to_coord(x1).detach().requires_grad_(True)
        score = self.score_net(y1, y0)
        div = partial_divergence(score, y1, create_graph=True).sum(dim=-1)
        return (div + 0.5 * score.pow(2).sum(dim=-1)).mean()

    def coefficient(self, t, x0, xt):
        y0 = self.to_coord(x0).detach()
        yt = self.to_coord(xt).detach()
        return self.lam_net(t.detach(), y0, yt)

    def assemble(self, tf_y, lam, t):
        """ Ambient control variate λ TF, bin-centered in t.

        Returns the centered control variate and the uncentered one.
        """
        raw = lam * self.to_ambient(tf_y)
        return bin_center(raw, t, self.n_bins), raw

    def bias_accumulator(self):
        return BiasAccumulator(self)


class BiasAccumulator:
    """ Epoch-level regression test of E[control | t, xt] = 0.

    The controller only sees (t, xt). Any conditional bias that can reach u_θ
    is a function of those, so the control variate is regressed on smooth
    features φ(t, y_t) = [1, t, y_t, t y_t, y_t²] accumulated over the epoch.
    Under the null every OLS coefficient is zero. zscore() returns the largest
    |β̂| / se(β̂) over coefficients and output dimensions. With ~160 entries the
    null maximum is about 3; values well above ~4.5 mean the control variate
    carries a conditional bias into u_θ.
    """

    def __init__(self, stein, n_bins=None):
        self.stein = stein
        self.G = None
        self.H = None
        self.M = None
        self.n = 0

    def _features(self, t, yt):
        return torch.cat([torch.ones_like(t), t, yt, t * yt, yt * yt], dim=-1)

    @torch.no_grad()
    def update(self, raw, t, xt):
        yt = self.stein.to_coord(xt)
        phi = self._features(t.detach(), yt).double()
        c = raw.double()
        if self.G is None:
            p, d = phi.shape[-1], c.shape[-1]
            self.G = phi.new_zeros(p, p)
            self.H = phi.new_zeros(p, d)
            self.M = phi.new_zeros(d, p, p)
        self.G += phi.T @ phi
        self.H += phi.T @ c
        # Heteroscedasticity-robust (White) meat, with c² in place of the
        # squared residual, which is exact under the null β = 0.
        self.M += torch.einsum("nd,np,nq->dpq", c * c, phi, phi)
        self.n += c.shape[0]

    @torch.no_grad()
    def zscore(self):
        if self.G is None or self.n < 10 * self.G.shape[0]:
            return 0.0
        p = self.G.shape[0]
        G = self.G + 1e-8 * torch.eye(p, dtype=self.G.dtype, device=self.G.device)
        Ginv = torch.linalg.inv(G)
        beta = Ginv @ self.H                                   # (p, d)
        cov = Ginv.unsqueeze(0) @ self.M @ Ginv.unsqueeze(0)   # (d, p, p)
        se = torch.diagonal(cov, dim1=-2, dim2=-1).clamp(min=0.0).sqrt().T  # (p, d)
        z = beta.abs() / se.clamp(min=1e-12)
        return float(z.max())


class DiagonalSteinNet(nn.Module):
    def __init__(self, dim, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, e):
        return self.net(torch.cat([x, e], dim=-1))


def stein_vector_diag(f_phi, energy, x, create_graph):
    x = x.detach().requires_grad_(True)
    e = energy.eval(x).unsqueeze(-1)
    f = f_phi(x, e)
    forces = energy(x)["forces"]

    div_diag = []
    for i in range(x.shape[-1]):
        df_i = torch.autograd.grad(
            f[:, i].sum(), x,
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, i]
        div_diag.append(df_i)
    div_diag = torch.stack(div_diag, dim=-1)
    tf = div_diag - f * forces
    return tf, f, forces


def fit_lambda(grad_E, tf, lam_max=10.0, eps=1e-8):
    g = grad_E - grad_E.mean(dim=0)
    t = tf - tf.mean(dim=0)
    lam = (g * t).mean(dim=0) / ((t * t).mean(dim=0) + eps)
    return lam.clamp(-lam_max, lam_max)
