# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

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
    """ One SGD step. Restore the weights if the step is not finite.

    Gradient clipping bounds the gradient. With SGD the parameter step is at
    most learning-rate times that bound, so a single batch cannot throw the
    weights out of range the way Adam does when its second moment is tiny.
    """
    if not torch.isfinite(loss).all():
        stein_opt.zero_grad(set_to_none=True)
        return False
    saved = [p.detach().clone() for p in params]
    stein_opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    if not torch.isfinite(grad_norm):
        stein_opt.zero_grad(set_to_none=True)
        return False
    stein_opt.step()
    if any(not torch.isfinite(p).all() for p in params):
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
        return False
    return True


def _restore(params, saved):
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)


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

    The buffer is filled exactly as in vanilla ASBS. Each gradient step takes
    one Hyvärinen step on the endpoint potential, then updates u_θ, then fits
    the test field and λ on the residual with u_θ held fixed. The potential
    never receives a gradient from the adjoint residual.

    A non-finite or exploded control variate is dropped for that batch, and
    u_θ is trained on the vanilla adjoint target. The CV loss penalizes
    ||λ TF||² equally with the residual, which shrinks a weak or biased
    correction back toward that same vanilla target.

    For the first `stein_warmup_epochs` the target is the vanilla ASBS one and
    only the potential trains.
    """
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale
    stein = matcher.stein
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
    score_params = stein.score_parameters()
    cv_params = stein.cv_parameters()

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        data = next(loader)
        pack = matcher.stein_batch(data, device, with_field=not warmup)
        t, xt = pack["t"], pack["xt"]
        adjoint = pack["adjoint"]

        pre = stein.hyvarinen(pack["x0"], pack["x1"])
        score_loss = pre.detach()
        # A positive Hyvärinen loss means ||score||² has overrun the divergence.
        # That is the blow-up seen in training, so the step is refused.
        if torch.isfinite(pre).all() and float(score_loss) < 1.0:
            saved = [p.detach().clone() for p in score_params]
            if _stein_step(stein_opt, score_params, pre):
                post = stein.hyvarinen(pack["x0"], pack["x1"])
                post_v = float(post.detach())
                del post
                pre_v = float(score_loss)
                if (not math.isfinite(post_v)) or post_v > max(pre_v + 5.0, 1.0):
                    _restore(score_params, saved)
                else:
                    score_loss = pre.new_tensor(post_v)

        if warmup:
            control = torch.zeros_like(adjoint)
            raw = control
            lam = control.new_zeros(1)
            use_cv = False
        else:
            tf_y = pack["tf_y"]
            lam = stein.coefficient(t, pack["x0"], xt)
            control, raw = stein.assemble(tf_y, lam, t)
            control_ms = control.detach().pow(2).mean()
            use_cv = bool(
                torch.isfinite(control).all()
                and torch.isfinite(control_ms)
                and float(control_ms) < 1e6
            )
            if not use_cv:
                control = torch.zeros_like(adjoint)
                raw = control

        epoch_score.update(score_loss.detach())

        optimizer.zero_grad(set_to_none=True)
        output = model(t, xt)
        target = -(adjoint - control.detach())
        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()
        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)
        optimizer.step()

        if use_cv:
            resid = output.detach() + adjoint - control
            # ||C||² with the same weight as the residual pulls λ halfway
            # back to 0. Vanilla ASBS is that limit.
            cv_loss = loss_scale * (resid.pow(2).mean() + control.pow(2).mean())
            _stein_step(stein_opt, cv_params, cv_loss)

            with torch.no_grad():
                corrected = adjoint - control.detach()
                var_ratio = corrected.pow(2).mean() / (adjoint.pow(2).mean() + 1e-8)
                epoch_var_ratio.update(var_ratio)
                epoch_lam.update(lam.detach().abs().mean())
                bias.update(raw.detach(), t, xt)
                epoch_tau.update(pack["tau"].mean())
        elif not warmup:
            epoch_var_ratio.update(adjoint.new_tensor(1.0))
            epoch_lam.update(adjoint.new_tensor(0.0))
            epoch_tau.update(pack["tau"].mean())
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
