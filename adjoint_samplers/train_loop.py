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
    """ Adjoint-matching epoch whose regression target includes a Stein CV.

    φ and λ(t) are updated on the same residual, with u_θ held fixed.
    The buffer itself is filled exactly as in vanilla ASBS.
    """
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
    epoch_var_ratio = MeanMetric().to(device, non_blocking=True)
    epoch_lam = MeanMetric().to(device, non_blocking=True)
    epoch_tf = MeanMetric().to(device, non_blocking=True)
    epoch_tau = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()
        data = next(loader)

        (t, xt), target, adjoint1, tf, lam, tau = matcher.prepare_target(
            data, device, create_graph=True,
        )
        output = model(t, xt)

        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()
        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)
        optimizer.step()

        with torch.no_grad():
            corrected = adjoint1 - lam * tf
            var_ratio = corrected.pow(2).mean() / (adjoint1.pow(2).mean() + 1e-8)
            epoch_var_ratio.update(var_ratio)
            epoch_lam.update(lam.abs().mean())
            epoch_tf.update(tf.mean(dim=0).norm())
            epoch_tau.update(tau.mean())

        # Same residual as the AM loss, but gradients flow only into φ and λ(t).
        cv_loss = ((output.detach() + adjoint1 - lam * tf) ** 2).mean()
        stein_opt.zero_grad()
        cv_loss.backward()
        stein_opt.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return {
        "loss": float(epoch_loss.compute().detach().cpu()),
        "stein_var_ratio": float(epoch_var_ratio.compute().detach().cpu()),
        "stein_lam_abs": float(epoch_lam.compute().detach().cpu()),
        "stein_tf_mean_norm": float(epoch_tf.compute().detach().cpu()),
        "stein_tau_mean": float(epoch_tau.compute().detach().cpu()),
    }
