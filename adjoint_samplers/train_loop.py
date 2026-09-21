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
    matcher, model, source, optimizer, lr_schedule, epoch, device, cfg,
    f_phi, stein_opt, energy,
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

        if "adjoint1" in data:  # only VE adjoint batches, not corrector-only
            from adjoint_samplers.components.stein_cv import stein_vector_diag
            import adjoint_samplers.utils.graph_utils as graph_utils

            x1 = data["x1"].to(device)
            TF, _, forces = stein_vector_diag(f_phi, energy, x1, create_graph=True)
            n, d = energy.n_particles, energy.n_spatial_dim
            TF = graph_utils.remove_mean(TF, n, d)
            target_cv = forces.detach() - forces.detach().mean(dim=0)
            cv_loss = ((TF - target_cv) ** 2).mean()
            stein_opt.zero_grad()
            cv_loss.backward()
            stein_opt.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()
            
    return float(epoch_loss.compute().detach().cpu())
