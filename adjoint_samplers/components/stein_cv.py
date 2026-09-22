import torch
import torch.nn as nn


class ContextSteinNet(nn.Module):
    """ Diagonal Stein test function f(x1, E(x1), t, x0, xt). """

    def __init__(self, dim, hidden=64, n_layers=3):
        super().__init__()
        in_dim = 3 * dim + 2
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x1, e, t, x0, xt):
        if e.ndim == 1:
            e = e.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x1, e, t, x0, xt], dim=-1))


class LambdaT(nn.Module):
    """ Scalar control-variate coefficient λ(t). """

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(t)


def ve_bridge_score_x1(ref_sde, t, x0, xt, x1, tau_max=0.99):
    """ ∇_{x1} log p_base(xt | x0, x1) for the VE Gaussian bridge.

    Returns the score and the unclamped reparametrized time τ(t).
    τ is clamped only inside the score so 1/(1-τ) stays finite.
    """
    total_var = ref_sde.total_var
    tau = ref_sde._diffsquare_integral(t) / total_var
    tau_c = tau.clamp(max=tau_max)
    s_br = (xt - (1 - tau_c) * x0 - tau_c * x1) / (total_var * (1 - tau_c))
    return s_br, tau


def stein_vector_bridge(f_phi, energy, t, x0, xt, x1, s_br, create_graph):
    """ Diagonal Stein field of f_φ against score s_br - ∇E.

    Divergence is only in x1. (t, x0, xt, s_br) are parameters.
    """
    x1 = x1.detach().requires_grad_(True)
    e = energy.eval(x1)
    if e.ndim == 1:
        e = e.unsqueeze(-1)
    f = f_phi(x1, e, t.detach(), x0.detach(), xt.detach())
    forces = energy(x1)["forces"].detach()
    score = s_br.detach() - forces

    div_diag = []
    for i in range(x1.shape[-1]):
        df_i = torch.autograd.grad(
            f[:, i].sum(),
            x1,
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, i]
        div_diag.append(df_i)
    div_diag = torch.stack(div_diag, dim=-1)
    tf = div_diag + f * score
    return tf, f, forces


class DiagonalSteinNet(nn.Module):
    def __init__(self, dim, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]  # f_phi(x, E) in R^d
        self.net = nn.Sequential(*layers)

    def forward(self, x, e):
        # x: (B, d), e: (B, 1)
        return self.net(torch.cat([x, e], dim=-1))

def stein_vector_diag(f_phi, energy, x, create_graph):
    x = x.detach().requires_grad_(True)
    e = energy.eval(x).unsqueeze(-1)          # (B, 1)
    f = f_phi(x, e)                           # (B, d)
    forces = energy(x)["forces"]              # (B, d), true ∇E, do not clip

    # (TF)_i = d f_i / d x_i - f_i * dE / d x_i
    div_diag = []
    for i in range(x.shape[-1]):
        df_i = torch.autograd.grad(
            f[:, i].sum(), x,
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, i]
        div_diag.append(df_i)
    div_diag = torch.stack(div_diag, dim=-1)  # (B, d)
    TF = div_diag - f * forces
    return TF, f, forces

def fit_lambda(grad_E, TF, lam_max=10.0, eps=1e-8):
    # grad_E, TF: (B, d), no grad needed for lambda
    g = grad_E - grad_E.mean(dim=0)
    t = TF - TF.mean(dim=0)
    lam = (g * t).mean(dim=0) / ((t * t).mean(dim=0) + eps)  # (d,)
    return lam.clamp(-lam_max, lam_max)