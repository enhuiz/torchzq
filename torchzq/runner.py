import zouqi
import argparse
import tqdm
import time
import yaml
import sys
import os
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from collections import defaultdict
from functools import partial, cached_property
from collections import defaultdict
from itertools import islice
from enum import Enum
from abc import ABC, abstractmethod

from .version import __version__
from .typing import Optional, Scheduled, _Scheduled
from .saver import Checkpoint, Saver
from .scheduler import Scheduler
from .interrupt import graceful_interrupt_handler
from .utils import print_directory_tree, default_tuple
from .metric import Metrics


class Runner(ABC):
    args = argparse.Namespace()

    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        VAL = 2

    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        strict_loading: bool = False,
        runs_root: Path = Path("runs"),
        use_fp16: bool = False,
        ckpt: Path = None,
        lr: Scheduled = "1e-3",
        seed: int = 0,
        wandb_project: str = "",
        ckpt_namespace: str = "default",
        log_every: int = 10,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.args.version = self.version

    ##############
    # Read-onlys #
    ##############

    @property
    def name(self):
        return self.args.name

    @property
    def version(self):
        return f"{__name__}-{__version__}"

    @property
    def run_dir(self):
        return self.args.runs_root / self.name

    @property
    def ckpt_dir(self):
        return self.run_dir / "ckpts"

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast, enabled=self.args.use_fp16)

    @cached_property
    def state(self):
        return self.init_ckpt.load_state()

    @property
    def global_step(self):
        return self.state.step

    @property
    def current_epoch(self):
        return self.state.epoch

    @property
    def Optimizer(self):
        return torch.optim.Adam

    @property
    def lr_coef(self):
        return 1

    ############
    # Settable #
    ############

    @property
    def backward_counter(self):
        return getattr(self, "_backward_counter", 0)

    @backward_counter.setter
    def backward_counter(self, value):
        self._backward_counter = value

    ########
    # Lazy #
    ########

    @cached_property
    def logger(self):
        args = self.args
        wandb.init(config=args, project=args.wandb_project, group=self.name)
        return wandb

    @cached_property
    def saver(self):
        return Saver(self.ckpt_dir)

    @cached_property
    def scheduler(self):
        scheduler = Scheduler()
        for k, v in list(vars(self.args).items()):
            if isinstance(v, _Scheduled):
                setattr(self.args, k, scheduler.schedule(v))
        return scheduler

    @cached_property
    def training_dataloader(self):
        return self.create_dataloader(self.Mode.TRAIN)

    @cached_property
    def validation_dataloader(self):
        return self.create_dataloader(self.Mode.VAL)

    @cached_property
    def testing_dataloader(self):
        return self.create_dataloader(self.Mode.TEST)

    @cached_property
    def sanity_check_dataloader(self):
        return list(islice(self.validation_dataloader, 3))

    @cached_property
    def metrics(self):
        metrics = self.create_metrics()
        self.init_ckpt.load_metrics(metrics)
        return metrics

    @cached_property
    def init_ckpt(self):
        args = self.args
        if args.ckpt is None:
            ckpt = self.saver.get(args.ckpt_namespace, Checkpoint())
        else:
            ckpt = Checkpoint.from_path(args.ckpt)
        return ckpt

    @cached_property
    def model(self):
        args = self.args
        scheduler = self.scheduler
        model = self.create_model()
        model.to(args.device)
        self.init_ckpt.load_model(model, args.strict_loading)
        scheduler.step(
            current_epoch=self.state.epoch,
            global_step=self.state.step,
        )
        return model

    @cached_property
    def optimizers(self):
        args = self.args
        optimizers = self.create_optimizers()
        if len(optimizers) == 0:
            raise ValueError("There should be at least 1 optimizer but get 0.")
        for optimizer in optimizers:
            for d in optimizer.state.values():
                for k, v in d.items():
                    if torch.is_tensor(v):
                        d[k] = v.to(args.device)
        self.init_ckpt.load_optimizers(optimizers)
        return optimizers

    @cached_property
    def scaler(self):
        args = self.args
        if args.use_fp16:
            scaler = torch.cuda.amp.GradScaler()
            self.init_ckpt.load_scaler(scaler)
        else:
            scaler = None
        return scaler

    ############
    # Creators #
    ############

    @abstractmethod
    def create_dataloader(self, mode: Mode):
        pass

    def create_metrics(self):
        return Metrics()

    def create_model(self):
        raise NotImplementedError

    def create_optimizers(self):
        self.scheduler
        args = self.args
        optimizer = self.Optimizer(params=self.model.parameters(), lr=1)
        args.lr.add_listener(lambda lr: self.set_lr(optimizer, self.lr_coef * lr))
        return [optimizer]

    #########
    # Misc. #
    #########

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def prepare_batch(self, batch, mode):
        args = self.args

        if isinstance(batch, dict):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

        if isinstance(batch, (tuple, list)):
            batch = list(batch)
            for i in range(len(batch)):
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(args.device)

        return batch

    def clip_grad_norm(self, optimizer_idx):
        args = self.args
        msg = "If you have more than one optimizers, please override clip_grad_norm()."
        assert optimizer_idx < 1, msg
        return nn.utils.clip_grad_norm_(
            self.model.parameters(),
            args.grad_clip_thres or 1e9,
        )

    #########
    # Steps #
    #########

    @abstractmethod
    def training_step(self, batch, optimizer_idx: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def training_step_with_optimization(self, batch) -> dict:
        args = self.args
        state = self.state

        stat_dict = {}
        start_time = time.time()

        self.backward_counter += 1
        state.step += self.backward_counter % args.update_every_backwards == 0

        self.scheduler.step(
            current_epoch=self.current_epoch,
            global_step=self.global_step,
        )

        for i, optimizer in enumerate(self.optimizers):
            with self.autocast_if_use_fp16():
                outputs = default_tuple(self.training_step(batch, i), [None, {}])

            loss = outputs[0]
            stat_dict.update(outputs[1])

            if loss is not None:
                if args.use_fp16:
                    self.scaler.scale(loss / args.update_every_backwards).backward()
                else:
                    (loss / args.update_every_backwards).backward()

            if self.backward_counter % args.update_every_backwards == 0:
                if args.use_fp16:
                    self.scaler.unscale_(optimizer)

                grad_norm = self.clip_grad_norm(i)

                stat_dict[f"grad_norm_{i}"] = grad_norm.item()

                if grad_norm.isfinite():
                    if args.use_fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                else:
                    print("Warning: grad_norm is not finite. Skip optimization.")

                optimizer.zero_grad()

        stat_dict["elapsed_time"] = time.time() - start_time

        return stat_dict

    def validation_step(self, batch, batch_idx: int) -> dict:
        stat_dict = {}
        for i in range(len(self.optimizers)):
            outputs = default_tuple(self.training_step(batch, i), [None, {}])
            stat_dict.update(outputs[1])
        return stat_dict

    @abstractmethod
    def testing_step(self, batch, batch_idx: int) -> dict:
        raise NotImplementedError

    def exit(self, reason):
        self.saver.buffer(
            self.state,
            self.model,
            self.optimizers,
            self.scaler,
            self.metrics,
            reason,
        )
        self.saver.dump_all()
        sys.exit(0)

    #########
    # Loops #
    #########

    def training_loop(self):
        args = self.args

        logger = self.logger

        # run sanity check before every training loop
        sanity_stat_dict = self.validate(sanity=True)

        # run metrics once before loop for sanity checking and state restoring
        # pass None to avoid updating the original value
        self.metrics({k: None for k in sanity_stat_dict})
        log_dict = self.metrics.to_dict()
        log_dict["epoch"] = self.current_epoch
        logger.log(log_dict, self.global_step)

        model = self.model.train()
        state = self.state

        def interrupt_callback(signum):
            print("Trying to gracefully shutdown ...")
            self.exit("interrupt")

        with graceful_interrupt_handler(callback=interrupt_callback):
            while self.current_epoch < args.max_epochs:
                state.epoch += 1
                desc = f"Train: {self.current_epoch}/{args.max_epochs}"
                pbar = tqdm.tqdm(self.training_dataloader, desc=desc)

                for local_step, batch in enumerate(pbar):
                    prepared_batch = self.prepare_batch(batch, self.Mode.TRAIN)

                    try:
                        stat_dict = self.training_step_with_optimization(prepared_batch)
                        if self.global_step % args.log_every == 0:
                            stat_dict = {f"train/{k}": v for k, v in stat_dict.items()}
                            logger.log(stat_dict, self.global_step)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            pbar.set_description("OOM! Skip batch.")
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad  # free some memory
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                    is_validating = (
                        args.validate_every_steps is not None
                        and self.global_step % args.validate_every_steps == 0
                    ) or (
                        args.validate_every_epochs is not None
                        and local_step == pbar.total - 1
                        and self.current_epoch % args.validate_every_epochs == 0
                    )

                    if is_validating:
                        self.metrics(self.validate())
                        log_dict = self.metrics.to_dict()
                        log_dict["epoch"] = self.current_epoch
                        logger.log(log_dict, self.global_step)
                        model.train()

                    is_saving = (
                        args.save_every_steps is not None
                        and self.global_step % args.save_every_steps == 0
                    ) or (
                        args.save_every_epochs is not None
                        and local_step == pbar.total - 1
                        and self.current_epoch % args.save_every_epochs == 0
                    )

                    if is_saving:
                        self.saver.buffer(
                            self.state,
                            model,
                            self.optimizers,
                            self.scaler,
                            self.metrics,
                        )
                        self.saver.dump()

                self.saver.buffer(
                    self.state,
                    model,
                    self.optimizers,
                    self.scaler,
                    self.metrics,
                )

        self.exit("finished")

    def val_test_loop(self, desc, dataloader, step_fn, mode):
        self.model.eval()
        pbar = tqdm.tqdm(dataloader, desc=desc)
        stats_dict = defaultdict(list)
        for index, batch in enumerate(pbar):
            prepared_batch = self.prepare_batch(batch, mode)
            outputs = default_tuple(step_fn(prepared_batch, index), [{}])
            for k, v in outputs[0].items():
                stats_dict[k].append(v)
        stat_dict = {k: np.mean(v) for k, v in stats_dict.items()}
        return stat_dict

    ############
    # Commands #
    ############

    @zouqi.command
    def train(
        self,
        weight_decay: float = 0,
        max_epochs: int = 100,
        validate_every_epochs: Optional[int] = 1,
        validate_every_steps: Optional[int] = None,
        save_every_epochs: Optional[int] = 1,
        save_every_steps: Optional[int] = None,
        update_every_backwards: int = 1,
        grad_clip_thres: float = 1.0,
    ):
        args = self.args

        tmpl = "{} not set, are you sure to skip {}? (y/n): "

        if save_every_steps is not None and save_every_epochs is not None:
            print(
                "Warning: save_every_steps and save_every_epochs are both set. "
                "Only save by steps."
            )

        if validate_every_epochs is None and validate_every_steps is None:
            if input(tmpl.format("validate_every_*", "validation")) != "y":
                exit()

        if save_every_epochs is None and save_every_steps is None:
            if input(tmpl.format("save_every_*", "saving")) != "y":
                exit()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        time_str = time.strftime("%Y%m%dT%H%M%S")
        with open(self.run_dir / f"config-{time_str}.yml", "w") as f:
            yaml.dump(vars(args), f)

        return self.training_loop()

    @zouqi.command
    def validate(self, sanity: bool = False):
        stat_dict = self.val_test_loop(
            f"Validating epoch {self.current_epoch} "
            + ("(sanity checking) " if sanity else "")
            + "...",
            self.sanity_check_dataloader if sanity else self.validation_dataloader,
            self.validation_step,
            self.Mode.VAL,
        )
        stat_dict = {f"val/{k}": v for k, v in stat_dict.items()}
        if stat_dict:
            stat_dict["epoch"] = self.current_epoch
            self.logger.log(stat_dict, self.global_step)
        return stat_dict

    @zouqi.command
    def test(self):
        stat_dict = self.val_test_loop(
            f"Testing epoch {self.current_epoch} ...",
            self.testing_dataloader,
            self.testing_step,
            self.Mode.TEST,
        )
        stat_dict = {f"test/{k}": v for k, v in stat_dict.items()}
        if stat_dict:
            stat_dict["epoch"] = self.current_epoch
            self.logger.log(stat_dict, self.global_step)
        return stat_dict

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    @zouqi.command
    def clear(self):
        args = self.args
        self.ls()
        if (args.runs_root / self.name).exists():
            if input("Are you sure to clear? (y)\n").lower() == "y":
                self.try_rmtree(args.runs_root / self.name)
            else:
                print(f"Not cleared.")
        else:
            print("Nothing to clear.")

    @zouqi.command
    def ls(self):
        args = self.args
        print_directory_tree(args.runs_root / self.name)
