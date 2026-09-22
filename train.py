# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import traceback
import hydra
import numpy as np
import termcolor

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_cv import LambdaT
from adjoint_samplers.train_loop import train_one_epoch, train_one_epoch_stein
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode


cudnn.benchmark = True


def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):

    try:
        train_utils.setup(cfg)
        print(str(cfg))

        device = "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Instantiating energy...")
        energy = hydra.utils.instantiate(cfg.energy, device=device)


        print("Instantiating source...")
        source = hydra.utils.instantiate(cfg.source, device=device)


        print('Instantiating model...')
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        controller = hydra.utils.instantiate(cfg.controller).to(device)
        sde = ControlledSDE(ref_sde, controller).to(device)


        if "corrector" in cfg:
            print('Instantiating corrector & corrector matcher...')
            corrector = hydra.utils.instantiate(cfg.corrector).to(device)
            corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)
        else:
            corrector = corrector_matcher = None

        f_phi = None
        lam_t = None
        if "stein_cv" in cfg:
            print("Instantiating Stein CV...")
            f_phi = hydra.utils.instantiate(cfg.stein_cv).to(device)
            lam_t = LambdaT().to(device)

        print("Instantiating grad of costs...")
        grad_term_cost = hydra.utils.instantiate(
            cfg.term_cost,
            corrector=corrector,
            energy=energy,
            ref_sde=ref_sde,
            source=source,
        )

        print("Instantiating adjoint matcher...")
        matcher_kwargs = {}
        if f_phi is not None:
            matcher_kwargs = dict(f_phi=f_phi, lam_t=lam_t, energy=energy)
        adjoint_matcher = hydra.utils.instantiate(
            cfg.adjoint_matcher,
            grad_term_cost=grad_term_cost,
            sde=sde,
            **matcher_kwargs,
        )


        print("Instantiating optimizer...")
        lr_schedule = None # TODO(ghliu) add scheduler
        if corrector is not None:
            optimizer = torch.optim.Adam([
                {'params': controller.parameters(), **cfg.adjoint_matcher.optim},
                {'params': corrector.parameters(), **cfg.corrector_matcher.optim},
            ])
        else:
            optimizer = torch.optim.Adam(
                controller.parameters(), **cfg.adjoint_matcher.optim,
            )

        stein_opt = None
        if f_phi is not None:
            stein_opt = torch.optim.Adam(
                list(f_phi.parameters()) + list(lam_t.parameters()),
                lr=1e-4,
            )

        checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
        checkpoint_path.parent.mkdir(exist_ok=True)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = train_utils.load(
                checkpoint,
                optimizer,
                controller,
                adjoint_matcher,
                corrector=corrector,
                corrector_matcher=corrector_matcher,
                f_phi=f_phi,
                lam_t=lam_t,
                stein_opt=stein_opt,
            )
            # Note: Not wrapping this in a DDP since we don't differentiate through SDE simulation.
        else:
            start_epoch = 0


        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )
            if corrector is not None:
                corrector = torch.nn.parallel.DistributedDataParallel(
                    corrector, device_ids=[cfg.gpu], find_unused_parameters=True
                )


        print("Instantiating writer...")
        writer = train_utils.Writer(
            name=cfg.exp_name,
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )


        print("Instantiating evaluator...")
        eval_dir = Path("eval_figs")
        eval_dir.mkdir(exist_ok=True)
        evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)


        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")
        for epoch in range(start_epoch, cfg.num_epochs):
            stage = train_utils.determine_stage(epoch, cfg)

            matcher, model = {
                "adjoint": (adjoint_matcher, controller),
                "corrector": (corrector_matcher, corrector),
            }.get(stage)

            log_extra = {}
            if f_phi is not None and stage == "adjoint":
                metrics = train_one_epoch_stein(
                    matcher, model, source, optimizer, lr_schedule,
                    epoch, device, cfg,
                    stein_opt=stein_opt,
                )
                loss = metrics["loss"]
                log_extra = {k: v for k, v in metrics.items() if k != "loss"}
            else:
                loss = train_one_epoch(
                    matcher, model, source, optimizer, lr_schedule,
                    epoch, device, cfg,
                )

            writer.log({
                f"{stage}_loss": loss,
                f"{stage}_buffer_size": len(matcher.buffer),
                **log_extra,
            }, step=epoch)

            extra_txt = ""
            if "stein_var_ratio" in log_extra:
                extra_txt = " var_ratio={:.3f} lam={:.3f}".format(
                    log_extra["stein_var_ratio"],
                    log_extra["stein_lam_abs"],
                )
            print("[{0} | {1}] {2}{3}".format(
                cyan(  f"{stage:<7}"),
                yellow(f"ep={epoch:04}"),
                green( f"loss={loss:.4f}"),
                extra_txt,
            ))

            # Eval epoch according to the frequency
            # otherwise eval at the end of adjoint matching
            if "eval_freq" in cfg:
                eval_this_epoch = epoch % cfg.eval_freq == 0
            else:
                eval_this_epoch = train_utils.is_last_am_epoch(epoch, cfg)

            if distributed_mode.is_main_process() and eval_this_epoch:
                # eval only after adjoint training
                if stage == "adjoint":
                    n_gen_samples = 0
                    x1_list = []
                    while n_gen_samples < cfg.num_eval_samples:
                        B = min(cfg.eval_batch_size, cfg.num_eval_samples - n_gen_samples)
                        x0 = source.sample([B,]).to(device)
                        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)

                        # model samples
                        x0, x1 = sdeint(
                            sde,
                            x0,
                            timesteps,
                            only_boundary=True,
                        )
                        x1_list.append(x1)
                        n_gen_samples += x1.shape[0]
                        print("Generated {} samples (total: {}/{})".format(
                            x1.shape[0],
                            n_gen_samples,
                            cfg.num_eval_samples,
                        ))

                    samples = torch.cat(x1_list, dim=0)
                    eval_dict = evaluator(samples)
                    print(f"Evaluated @{epoch=}!")

                    if "hist_img" in eval_dict:
                        eval_dict["hist_img"].save(eval_dir / "gen.png")

                    writer.log(eval_dict, step=epoch)

                print("Saving checkpoint ... ")
                train_utils.save(
                    epoch,
                    cfg,
                    optimizer,
                    controller,
                    adjoint_matcher,
                    corrector=corrector,
                    corrector_matcher=corrector_matcher,
                    f_phi=f_phi,
                    lam_t=lam_t,
                    stein_opt=stein_opt,
                )

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
