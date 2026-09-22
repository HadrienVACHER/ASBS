# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import math
from typing import Any
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import PIL
import wandb

import torch
from torch.optim import Optimizer

import adjoint_samplers.utils.distributed_mode as distributed_mode
from adjoint_samplers.components.matcher import Matcher


def setup(cfg):
    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(
                props.name, props.total_memory / (1024**3)
            )
        )
    print(dict(os.environ))
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    distributed_mode.init_distributed_mode(cfg)

    if distributed_mode.is_main_process():
        args_filepath = Path("cfg.yaml")
        print(f"Saving cfg to {args_filepath}")
        with open("config.yaml", "w") as fout:
            print(OmegaConf.to_yaml(cfg), file=fout)
        with open("env.json", "w") as fout:
            print(json.dumps(dict(os.environ)), file=fout)


def get_timesteps(
    t0: torch.Tensor | float,
    t1: torch.Tensor | float,
    dt: torch.Tensor | float | None = None,
    steps: int | None = None,
    rescale_t: str | None = None
) -> torch.Tensor:
    if (steps is None) is (dt is None):
        raise ValueError("Exactly one of `dt` and `steps` should be defined.")
    if steps is None:
        steps = int(math.ceil((t1 - t0) / dt))
    if rescale_t is None:
        return torch.linspace(t0, t1, steps=steps)
    elif rescale_t == "quad":
        return torch.sqrt(
            torch.linspace(t0, t1.square(), steps=steps)
        ).clip(max=t1)
    elif rescale_t == "cosine":
        """
        from https://github.com/franciscovargas/denoising_diffusion_samplers/blob/main/dds/discretisation_schemes.py
        """
        s = 0.008  # Choice from original paper
        pre_phase = torch.linspace(t0, t1, steps) / t1
        phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5

        dts = torch.cos(phase) ** 4

        dts /= dts.sum()
        dts *= t1  # We normalise s.t. \sum_k \beta_k = T (where beta_k = b_m*cos^4)

        dts_out = torch.concat(
            (torch.tensor([t0]), torch.cumsum(dts, -1))
        )

        return dts_out
    raise ValueError("Unkown timestep rescaling method.")


def is_asbs_init_stage(epoch: int, cfg: DictConfig):
    if "corrector" not in cfg:
        return False

    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_c = cfg.corrector_matcher.num_epochs_per_stage

    assert cfg.init_stage in ("adjoint", "corrector")
    n = n_a if cfg.init_stage == "adjoint" else n_c
    return epoch < n


def determine_stage(epoch: int, cfg: DictConfig):
    if "corrector" not in cfg:
        return "adjoint"

    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_c = cfg.corrector_matcher.num_epochs_per_stage

    if cfg.init_stage == "adjoint":
        return "adjoint" if (epoch % (n_a + n_c)) < n_a else "corrector"

    elif cfg.init_stage == "corrector":
        return "corrector" if (epoch % (n_a + n_c)) < n_c else "adjoint"
    else:
        raise NotImplementedError


def is_last_am_epoch(epoch: int, cfg: DictConfig):
    if "corrector" not in cfg:
        return cfg.adjoint_matcher.num_epochs_per_stage

    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_c = cfg.corrector_matcher.num_epochs_per_stage

    assert cfg.init_stage in ("adjoint", "corrector")
    if cfg.init_stage == "adjoint":
        return (epoch + 1) % (n_a + n_c) == n_a
    else:
        return (epoch + 1) % (n_a + n_c) == 0


def save(
    epoch: int,
    cfg: DictConfig,
    optimizer: Optimizer,
    controller: torch.nn.Module,
    adjoint_matcher: Matcher,
    corrector: torch.nn.Module | None = None,
    corrector_matcher: Matcher | None = None,
    ckpt_dir: Path = Path("checkpoints"),
    f_phi: torch.nn.Module | None = None,
    lam_t: torch.nn.Module | None = None,
    stein_opt: Optimizer | None = None,
):
    ckpt_dir.mkdir(exist_ok=True)

    state = {
        "epoch": epoch,
        "cfg": cfg,
        "optimizer": optimizer.state_dict(),
    }

    def get_state_dict(module):
        if cfg.distributed and hasattr(module, "module"):
            return module.module.state_dict()
        else:
            return module.state_dict()
    state["controller"] = get_state_dict(controller)
    if corrector is not None:
        state["corrector"] = get_state_dict(corrector)
    if f_phi is not None:
        state["f_phi"] = f_phi.state_dict()
    if lam_t is not None:
        state["lam_t"] = lam_t.state_dict()
    if stein_opt is not None:
        state["stein_opt"] = stein_opt.state_dict()

    # Save current checkpoint
    torch.save(state, ckpt_dir / "checkpoint_{}.pt".format(epoch))

    # Save latest checkpoint with buffer
    state["adjoint_buffer"] = adjoint_matcher.buffer.state_dict()
    if corrector_matcher is not None:
        state["corrector_buffer"] = corrector_matcher.buffer.state_dict()
    torch.save(state, ckpt_dir / "checkpoint_latest.pt")


def load(
    checkpoint: dict[str: Any],
    optimizer: Optimizer,
    controller: torch.nn.Module,
    adjoint_matcher: Matcher,
    corrector: torch.nn.Module | None = None,
    corrector_matcher: Matcher | None = None,
    f_phi: torch.nn.Module | None = None,
    lam_t: torch.nn.Module | None = None,
    stein_opt: Optimizer | None = None,
):
    optimizer.load_state_dict(checkpoint["optimizer"])
    controller.load_state_dict(checkpoint["controller"])

    if "adjoint_buffer" in checkpoint:
        adjoint_matcher.buffer.load_state_dict(checkpoint["adjoint_buffer"])

    if corrector is not None and "corrector" in checkpoint:
        corrector.load_state_dict(checkpoint["corrector"])

    if corrector_matcher is not None and "corrector_buffer" in checkpoint:
        corrector_matcher.buffer.load_state_dict(checkpoint["corrector_buffer"])

    if f_phi is not None and "f_phi" in checkpoint:
        f_phi.load_state_dict(checkpoint["f_phi"])
    if lam_t is not None and "lam_t" in checkpoint:
        lam_t.load_state_dict(checkpoint["lam_t"])
    if stein_opt is not None and "stein_opt" in checkpoint:
        stein_opt.load_state_dict(checkpoint["stein_opt"])

    return checkpoint["epoch"] + 1


class Writer:
    def __init__(self, name: str, cfg: DictConfig, is_main_process: bool):
        if cfg.use_wandb and is_main_process:
            self.writer = wandb.init(
                mode='online',
                project=cfg.project,
                name=name,
                config=dict(cfg),
            )
        else:
            self.writer = None

    def log(self, log_dict: dict, step=None):
        if self.writer is not None:
            # convert images
            for k in log_dict.keys():
                if isinstance(log_dict[k], PIL.Image.Image):
                    log_dict[k] = wandb.Image(log_dict[k])
            self.writer.log(log_dict, step=step)
