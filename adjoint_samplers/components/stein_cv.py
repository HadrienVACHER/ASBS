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

    The endpoint conditional score is the gradient of a scalar potential, which
    is the function class a score belongs to. λ(t, x0, xt) is bounded and the
    Stein residual is divided by a lagged root-mean-square, so the correction
    stays the same size as the adjoint instead of amplifying a score error.
    Both start at zero, and the first controller updates match vanilla ASBS.
    """

    def __init__(
        self,
        n_particles,
        spatial_dim,
        hidden=64,
        score_hidden=128,
        n_layers=3,
        tau_max=0.999,
        lam_max=1.0,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.tau_max = tau_max
        self.lam_max = lam_max
        rank = (n_particles - 1) * spatial_dim
        self.rank = rank
        self.register_buffer("Q", mean_free_basis(n_particles, spatial_dim))
        # 0 marks "no past scale yet". Division uses the previous value.
        self.register_buffer("tf_rms", torch.tensor(0.0))

        # λ starts at 0 so the first update equals vanilla ASBS. The field must
        # not start at 0: with TF = 0 and λ = 0 neither gets a gradient.
        self.field = ConcatMLP(3 * rank + 2, rank, hidden, n_layers)
        self.potential = ConcatMLP(2 * rank, 1, score_hidden, n_layers, zero_last=True)
        self.lam_net = ConcatMLP(1 + 2 * rank, 1, hidden, n_layers, zero_last=True)

    def score_parameters(self):
        return list(self.potential.parameters())

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
        score = s_br + self.endpoint_score(y1, y0)

        x1_var = self.to_ambient(y1)
        energy_value = energy.eval(x1_var)
        if energy_value.ndim == 1:
            energy_value = energy_value.unsqueeze(-1)
        # asinh keeps a quartic energy, and a far-away configuration, from
        # saturating the linear layers. It is a fixed function of this sample,
        # so the divergence in y1 is still a Stein divergence.
        field = self.field(
            torch.asinh(y1),
            torch.asinh(energy_value),
            t,
            torch.asinh(y0),
            torch.asinh(yt),
        )
        tf = partial_divergence(field, y1, create_graph=True) + field * score

        # (1-τ) is a function of t, so it preserves a conditional mean of zero.
        # It cancels the 1/(1-τ) pole of s_br.
        scale = (1.0 - tau).detach()
        tf = torch.where(active, tf * scale, torch.zeros_like(tf))
        return self._scale_tf(tf), tau.detach()

    def endpoint_score(self, y1, y0):
        """ ∇_{y1} ψ(y1, y0), detached from the potential parameters. """
        with torch.enable_grad():
            leaf = y1.detach().requires_grad_(True)
            psi = self.potential(leaf, y0.detach())
            score = torch.autograd.grad(psi.sum(), leaf, create_graph=False)[0]
        return score.detach()

    def _scale_tf(self, tf):
        """ Divide by the previous root-mean-square.

        The divisor is a lagged scalar, so a conditional mean of zero stays
        zero. After this, λ of order 1 is a correction of the same size as the
        residual rather than a gain of ten or twenty on a tiny field.
        """
        batch = tf.detach().pow(2).mean().sqrt()
        have_scale = float(self.tf_rms) > 0.0
        scale = float(self.tf_rms.clamp(min=1e-3)) if have_scale else 1.0
        scaled = tf / scale
        if torch.isfinite(batch).all():
            obs = float(batch.clamp(min=1e-3))
            if have_scale:
                obs = min(obs, scale * 10.0)
            with torch.no_grad():
                if have_scale:
                    self.tf_rms.mul_(0.99).add_(obs * 0.01)
                else:
                    self.tf_rms.fill_(obs)
        return scaled

    def hyvarinen(self, x0, x1):
        """ Conditional implicit score matching for ∇_{y1} log p(y1 | y0). """
        y0 = self.to_coord(x0).detach()
        y1 = self.to_coord(x1).detach().requires_grad_(True)
        psi = self.potential(y1, y0)
        score = torch.autograd.grad(psi.sum(), y1, create_graph=True)[0]
        div = partial_divergence(score, y1, create_graph=True).sum(dim=-1)
        return (div + 0.5 * score.pow(2).sum(dim=-1)).mean()

    def coefficient(self, t, x0, xt):
        y0 = torch.asinh(self.to_coord(x0).detach())
        yt = torch.asinh(self.to_coord(xt).detach())
        raw = self.lam_net(t.detach(), y0, yt)
        return self.lam_max * torch.tanh(raw)

    def assemble(self, tf_y, lam, t):
        """ Ambient control variate λ TF.

        Returns the control variate twice. The second copy is the one the
        bias regression sees; nothing is subtracted from it.
        """
        del t
        raw = lam * self.to_ambient(tf_y)
        return raw, raw

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
