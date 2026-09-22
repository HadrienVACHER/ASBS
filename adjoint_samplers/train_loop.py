# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_one_epoch(
    matcher: Matcher,
    model: torch.nn.Module,
    source: torch.nn.Module,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    epoch: int,
    device: str,
    cfg: DictConfig,
):
    # build dataloader
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()

        data = next(loader)

        input, target = matcher.prepare_target(data, device)
        output = model(*input)

        loss = loss_scale * ((output - target)**2).mean()
        loss.backward()

        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)

        optimizer.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return float(epoch_loss.compute().detach().cpu())


def _stein_step(stein_opt, params, loss):
    stein_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    stein_opt.step()


def train_one_epoch_stein(
    matcher,
    model,
    source,
    optimizer,
    lr_schedule,
    epoch,
    device,
    cfg,
    stein_opt,
):
    """ Adjoint-matching epoch with a subspace Stein control variate.

    The buffer is filled exactly as in vanilla ASBS. Each gradient step fits
    the endpoint score by Hyvärinen matching on buffer pairs, then updates
    u_θ on the control-variate target, then fits the test field and λ on the
    same residual with u_θ held fixed. The score network never receives a
    gradient from the adjoint residual.

    For the first `stein_warmup_epochs` the target is the vanilla ASBS one and
    only the score network trains.
    """
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale
    stein = matcher.stein
    score_steps = int(cfg.get("stein_score_steps", 2))
    warmup = epoch < int(cfg.get("stein_warmup_epochs", 0))

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_var_ratio = MeanMetric().to(device, non_blocking=True)
    epoch_lam = MeanMetric().to(device, non_blocking=True)
    bias = stein.bias_accumulator()
    epoch_score = MeanMetric().to(device, non_blocking=True)
    epoch_tau = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))
    score_params = list(stein.score_net.parameters())
    cv_params = stein.cv_parameters()

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        data = next(loader)
        pack = matcher.stein_batch(data, device, with_field=not warmup)
        t, xt = pack["t"], pack["xt"]
        adjoint = pack["adjoint"]

        score_loss = None
        for _ in range(score_steps):
            score_loss = stein.hyvarinen(pack["x0"], pack["x1"])
            _stein_step(stein_opt, score_params, score_loss)

        if warmup:
            control = torch.zeros_like(adjoint)
        else:
            tf_y = pack["tf_y"]
            lam = stein.coefficient(t, pack["x0"], xt)
            control, raw = stein.assemble(tf_y, lam, t)

        optimizer.zero_grad(set_to_none=True)
        output = model(t, xt)
        target = -(adjoint - control.detach())
        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()
        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)
        optimizer.step()

        if not warmup:
            cv_loss = loss_scale * ((output.detach() + adjoint - control) ** 2).mean()
            _stein_step(stein_opt, cv_params, cv_loss)

            with torch.no_grad():
                corrected = adjoint - control.detach()
                var_ratio = corrected.pow(2).mean() / (adjoint.pow(2).mean() + 1e-8)
                epoch_var_ratio.update(var_ratio)
                epoch_lam.update(lam.detach().abs().mean())
                bias.update(raw.detach(), t, xt)
                epoch_tau.update(pack["tau"].mean())

        epoch_score.update(score_loss.detach())
        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    def value(metric, default):
        if warmup and metric is not epoch_loss and metric is not epoch_score:
            return default
        return float(metric.compute().detach().cpu())

    return {
        "loss": value(epoch_loss, 0.0),
        "stein_var_ratio": value(epoch_var_ratio, 1.0),
        "stein_lam_abs": value(epoch_lam, 0.0),
        "stein_bias_z": bias.zscore(),
        "stein_score_loss": value(epoch_score, 0.0),
        "stein_tau_mean": value(epoch_tau, 0.0),
        "stein_active": 0.0 if warmup else 1.0,
    }
