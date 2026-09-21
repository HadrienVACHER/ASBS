import torch
import torch.nn as nn

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