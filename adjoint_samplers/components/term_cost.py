# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import adjoint_samplers.utils.graph_utils as graph_utils
from adjoint_samplers.components.stein_cv import stein_vector_diag, fit_lambda

class GradEnergy:
    """ Compute ∇E(X_1)
    """
    def __init__(self, energy, max_grad_E_norm = None, **kwargs):
        self.energy = energy
        self.max_grad_E_norm = max_grad_E_norm

    def clip(self, grad_E):
        if self.max_grad_E_norm is not None:
            norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
            clip_coefficient = torch.clamp(self.max_grad_E_norm / (norm + 1e-6), max=1)
            clip_coefficient = clip_coefficient.unsqueeze(-1)
        else:
            clip_coefficient = torch.ones_like(grad_E)

        return grad_E * clip_coefficient

    def grad_E(self, x1):
        grad_E = self.energy(x1)["forces"]
        return self.clip(grad_E)

    def __call__(self, x1):
        return self.grad_E(x1)


# For AS.
class ScoreGradTermCost(GradEnergy):
    """ Compute (∇E + ∇log p^base_1)(X_1)
    """
    def __init__(self, source, ref_sde, energy, **kwargs):
        super().__init__(energy, **kwargs)

        self._check_source_class(source)

        # Compute p^base_1 = N(x1; μ1, Σ1)
        mu0, var0 = source.loc, source.scale**2
        t1 = torch.ones(1, device=mu0.device)
        mu1, var1 = ref_sde._pt_gauss_param(t1, mu0, var0)
        self.mu1 = mu1.reshape(-1)
        self.var1 = var1.reshape(-1)

    def _check_source_class(self, source):
        from adjoint_samplers.utils.dist_utils import Delta, Gauss
        assert isinstance(source, (Delta, Gauss))

    def __call__(self, x1):
        # Compute ∇log p^base_1(x) = (μ1 - x) / Σ1
        score = (self.mu1.to(x1) - x1) / self.var1.to(x1)
        return self.grad_E(x1) + score


# For ASBS.
class CorrectorGradTermCost(GradEnergy):
    """ Compute (∇E + ∇log h)(X_1), where h is the corrector of ASBS.
    """
    def __init__(self, corrector, energy, **kwargs):
        super().__init__(energy, **kwargs)
        self.corrector = corrector

    def __call__(self, x1, **kwargs):
        t1 = torch.ones(x1.shape[0], 1).to(x1)
        with torch.no_grad():
            corrector = self.corrector(t1, x1)
        return self.grad_E(x1) + corrector


# For ASBS on n-particle systems.
class GraphCorrectorGradTermCost(CorrectorGradTermCost):
    def __init__(self, corrector, energy, **kwargs):
        super().__init__(corrector, energy, **kwargs)
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

    def grad_E(self, x1):
        N, D = self.n_particles, self.n_spatial_dim

        grad_E = self.energy(x1)["forces"]

        # clip spatial dim
        grad_E = self.clip(grad_E.view(-1, N, D)).view(-1, N * D)

        grad_E = graph_utils.remove_mean(grad_E, N, D)
        return grad_E


# For AS on n-particle systems.
class GraphScoreGradTermCost(ScoreGradTermCost):
    """ Compute (∇E + ∇log p^base_1)(X_1) for n-particle systems
    """
    def __init__(self, source, ref_sde, energy, **kwargs):
        super().__init__(source, ref_sde, energy, **kwargs)
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

    def _check_source_class(self, source):
        from adjoint_samplers.utils.dist_utils import Delta, CenteredParticlesGauss
        assert isinstance(source, (Delta, CenteredParticlesGauss))

    def grad_E(self, x1):
        N, D = self.n_particles, self.n_spatial_dim

        grad_E = self.energy(x1)["forces"]

        # clip spatial dim
        grad_E = self.clip(grad_E.view(-1, N, D)).view(-1, N * D)

        grad_E = graph_utils.remove_mean(grad_E, N, D)
        return grad_E

# For ASBS with Stein CV on n-particle systems.
class GraphSteinCVCorrectorGradTermCost(CorrectorGradTermCost):
    def __init__(self, corrector, energy, stein_cv=None, **kwargs):
        super().__init__(corrector, energy, **kwargs)
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim
        self.stein_cv = stein_cv  # None = current ASBS

    def cv_force(self, x1):
        raw = self.energy(x1)["forces"]
        if self.stein_cv is None:
            return raw
        TF, _, _ = stein_vector_diag(
            self.stein_cv, self.energy, x1, create_graph=False
        )
        TF = graph_utils.remove_mean(TF, self.n_particles, self.n_spatial_dim)
        lam = fit_lambda(raw.detach(), TF.detach())
        return raw - lam * TF

    def grad_E(self, x1):
        N, D = self.n_particles, self.n_spatial_dim
        grad_E = self.cv_force(x1)
        grad_E = self.clip(grad_E.view(-1, N, D)).view(-1, N * D)
        return graph_utils.remove_mean(grad_E, N, D)
