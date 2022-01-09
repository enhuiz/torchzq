import tqdm
import time
import yaml
import sys
import os
import gc
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from functools import partial, cached_property
from collections import defaultdict
from itertools import islice
from enum import Enum
from typing import Annotated, Optional
from argparse_hparams import HParams as HParamsBase
from dataclasses import dataclass

from .version import __version__
from .saver import Checkpoint, Saver
from .scheduler import Scheduler
from .interrupt import graceful_interrupt_handler
from .utils import make_grad_dataframe, print_directory_tree, flatten_dict
from .metric import Metrics


class _Scheduled(str):
    pass


Scheduled = Annotated[str, dict(type=_Scheduled)]


def command(func):
    func._is_command = True
    return func


class Runner:
    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        VAL = 2

    @dataclass
    class HParams(HParamsBase):
        command: Annotated[str, dict(positional=True)] = ""

        runner: str = ""
        version: str = ""
        name: str = "Unnamed"
        nj: int = min(os.cpu_count() or 12, 12)
        device: str = "cuda"
        strict_loading: bool = False
        runs_root: Path = Path("runs")
        use_fp16: bool = False
        ckpt: Optional[Path] = None
        pretrained_ckpt: Optional[Path] = None
        lr: Scheduled = "1e-3"
        seed: int = 0
        wandb_project: str = ""
        ckpt_namespace: str = "default"
        log_every: int = 10

        # train
        batch_size: int = 32
        weight_decay: float = 0
        max_epochs: int = 100
        validate_every_epochs: Optional[int] = 1
        validate_every_steps: Optional[int] = None
        save_every_epochs: Optional[int] = 1
        save_every_steps: Optional[int] = None
        update_every_backwards: int = 1
        grad_clip_thres: Optional[float] = 1.0
        grad_log_every: Optional[int] = 100

    hp: HParams

    def __init__(self):
        self.hp = self.HParams()
        random.seed(self.hp.seed)
        np.random.seed(self.hp.seed)
        torch.manual_seed(self.hp.seed)
        self.hp.version = self.version

    def start(self):
        self.hp.show(sort=True)
        command_fn = getattr(self, self.hp.command)
        if command_fn._is_command:
            command_fn()
        else:
            raise ValueError(f"{self.hp.command} is not a command.")

    ##############
    # Read-onlys #
    ##############

    @property
    def name(self):
        return self.hp.name

    @property
    def version(self):
        return f"{__name__}-{__version__}"

    @property
    def run_dir(self):
        return self.hp.runs_root / self.name

    @property
    def ckpt_dir(self):
        return self.run_dir / "ckpts"

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast_mode.autocast, enabled=self.hp.use_fp16)

    @cached_property
    def state(self):
        return self.starting_ckpt.load_state()

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
        wandb.init(config=self.hp, project=self.hp.wandb_project, group=self.name)
        return wandb

    @cached_property
    def saver(self):
        return Saver(self.ckpt_dir)

    @cached_property
    def scheduler(self):
        scheduler = Scheduler()
        for k, v in list(vars(self.hp).items()):
            if isinstance(v, _Scheduled):
                setattr(self.hp, k, scheduler.schedule(v))
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
        self.starting_ckpt.load_metrics(metrics)
        return metrics

    @cached_property
    def pretrained_ckpt(self):
        """
        An optional checkpoint loaded before any checkpoint.
        It can be overwritten by the starting_ckpt.
        """
        hp = self.hp
        if hp.pretrained_ckpt is None:
            ckpt = Checkpoint()
        else:
            ckpt = Checkpoint.from_path(hp.pretrained_ckpt)
        return ckpt

    @cached_property
    def starting_ckpt(self):
        """
        The checkpoint used when resuming training / validate / test.
        """
        hp = self.hp
        if hp.ckpt is None:
            ckpt = self.saver.get(hp.ckpt_namespace, Checkpoint())
        else:
            ckpt = Checkpoint.from_path(hp.ckpt)
        return ckpt

    @cached_property
    def model(self):
        hp = self.hp
        scheduler = self.scheduler
        model = self.create_model()
        self.pretrained_ckpt.load_model(model, hp.strict_loading)
        self.starting_ckpt.load_model(model, hp.strict_loading)
        model.to(hp.device)
        scheduler.step(
            current_epoch=self.state.epoch,
            global_step=self.state.step,
        )
        return model

    @cached_property
    def optimizers(self):
        hp = self.hp
        optimizers = self.create_optimizers()
        if len(optimizers) == 0:
            raise ValueError("There should be at least 1 optimizer but get 0.")
        for optimizer in optimizers:
            for d in optimizer.state.values():
                for k, v in d.items():
                    if torch.is_tensor(v):
                        d[k] = v.to(hp.device)
        self.starting_ckpt.load_optimizers(optimizers)
        return optimizers

    @cached_property
    def scaler(self):
        hp = self.hp
        if hp.use_fp16:
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
            self.starting_ckpt.load_scaler(scaler)
        else:
            scaler = None
        return scaler

    ############
    # Creators #
    ############

    def create_dataloader(self, mode: Mode) -> DataLoader:
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def create_metrics(self):
        return Metrics()

    def create_optimizers(self):
        self.scheduler
        hp = self.hp
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.Optimizer(params=params, lr=1)
        hp.lr.add_listener(lambda lr: self.set_lr(optimizer, self.lr_coef * lr))
        return [optimizer]

    #########
    # Misc. #
    #########
    @staticmethod
    def fill_missing(outputs, default):
        if outputs is None:
            outputs = []
        elif isinstance(outputs, tuple):
            outputs = list(outputs)
        else:
            outputs = [outputs]

        for i in range(len(outputs), len(default)):
            outputs.append(default[i])

        return tuple(outputs)

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def prepare_batch(self, batch, mode):
        hp = self.hp

        if isinstance(batch, dict):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(hp.device)

        if isinstance(batch, (tuple, list)):
            batch = list(batch)
            for i in range(len(batch)):
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(hp.device)

        return batch

    def clip_grad_norm(self, optimizer_idx):
        hp = self.hp
        msg = "Multiple optimizers detected, please override clip_grad_norm()."
        assert optimizer_idx < 1, msg
        return nn.utils.clip_grad_norm_(
            self.model.parameters(),
            hp.grad_clip_thres or 1e9,
        )

    #########
    # Steps #
    #########

    def training_step(self, batch, optimizer_idx: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def training_step_with_optimization(self, batch) -> dict:
        hp = self.hp
        state = self.state

        scalar_dict = {}
        start_time = time.time()

        self.backward_counter += 1
        state.step += self.backward_counter % hp.update_every_backwards == 0

        self.scheduler.step(
            current_epoch=self.current_epoch,
            global_step=self.global_step,
        )

        for i, optimizer in enumerate(self.optimizers):
            with self.autocast_if_use_fp16():
                outputs = self.fill_missing(self.training_step(batch, i), (None, {}))

            loss = outputs[0]
            scalar_dict.update(outputs[1])

            if loss is not None:
                if hp.use_fp16:
                    assert self.scaler is not None
                    self.scaler.scale(loss / hp.update_every_backwards).backward()
                else:
                    (loss / hp.update_every_backwards).backward()

            if self.backward_counter % hp.update_every_backwards == 0:
                if hp.use_fp16:
                    assert self.scaler is not None
                    self.scaler.unscale_(optimizer)

                if hp.grad_log_every and self.global_step % hp.grad_log_every == 0:
                    path = self.run_dir / "grad" / f"{self.global_step}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    make_grad_dataframe(self.model).to_csv(path, index=False)
                    del path

                grad_norm = self.clip_grad_norm(optimizer_idx=i)

                scalar_dict[f"grad_norm_{i}"] = grad_norm.item()

                if hp.use_fp16:
                    # scaler.step will automatically skip optimizer.step
                    # if grad_norm is not finite
                    assert self.scaler is not None
                    self.scaler.step(optimizer)
                    self.scaler.update()
                elif grad_norm.isfinite():
                    optimizer.step()

                if not grad_norm.isfinite():
                    print("Warning: grad_norm is not finite. Skip optimization.")

                optimizer.zero_grad()

        scalar_dict["elapsed_time"] = time.time() - start_time

        return scalar_dict

    def validation_step(self, batch, batch_idx: int) -> dict:
        scalar_dict = {}
        for i in range(len(self.optimizers)):
            outputs = self.fill_missing(self.training_step(batch, i), (None, {}))
            scalar_dict.update(outputs[1])
        return scalar_dict

    @property
    def testing_step(self):
        return self.validation_step

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

    @staticmethod
    def build_checker(every_step, every_epoch):
        # divisible (and exists)
        d = lambda a, b: b is not None and a % b == 0
        # divisible by step
        ds = lambda a: d(a, every_step)
        # divisible by epoch
        de = lambda a: d(a, every_epoch)
        # checker
        return lambda step, epoch, last: ds(step) or (de(epoch) and last)

    @cached_property
    def is_saving(self):
        hp = self.hp
        return self.build_checker(hp.save_every_steps, hp.save_every_epochs)

    @cached_property
    def is_validating(self):
        hp = self.hp
        return self.build_checker(hp.validate_every_steps, hp.validate_every_epochs)

    def training_loop(self):
        hp = self.hp
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

        def interrupt_callback(_):
            print("Trying to gracefully shutdown ...")
            self.exit("interrupt")

        with graceful_interrupt_handler(callback=interrupt_callback):
            while self.current_epoch < hp.max_epochs:
                state.epoch += 1
                desc = f"Train: {self.current_epoch}/{hp.max_epochs}"
                pbar = tqdm.tqdm(self.training_dataloader, desc=desc)
                total = len(self.training_dataloader)

                for local_step, batch in enumerate(pbar):
                    batch = self.prepare_batch(batch, self.Mode.TRAIN)

                    oom = False
                    try:
                        scalar_dict = self.training_step_with_optimization(batch)
                        if self.global_step % hp.log_every == 0:
                            scalar_dict = flatten_dict(dict(train=scalar_dict))
                            logger.log(scalar_dict, self.global_step)
                        del scalar_dict
                    except RuntimeError as e:
                        oom = "out of memory" in str(e)
                        if not oom:
                            raise e
                    finally:
                        if oom:
                            print("OOM! Skip batch.")
                            del batch
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad
                            gc.collect()

                    check = lambda checker: checker(
                        step=self.global_step,
                        epoch=self.current_epoch,
                        last=local_step == total - 1,
                    )

                    if check(self.is_validating):
                        self.metrics(self.validate())
                        log_dict = self.metrics.to_dict()
                        log_dict["epoch"] = self.current_epoch
                        logger.log(log_dict, self.global_step)
                        model.train()

                    if check(self.is_saving):
                        self.saver.buffer(
                            self.state,
                            self.model,
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
            batch = self.prepare_batch(batch, mode)
            outputs = self.fill_missing(step_fn(batch, index), ({},))
            for k, v in outputs[0].items():
                stats_dict[k].append(v)
        scalar_dict = {k: np.mean(v) for k, v in stats_dict.items()}
        return scalar_dict

    ############
    # Commands #
    ############

    @command
    def train(self):
        hp = self.hp

        tmpl = "{} not set, are you sure to skip {}? (y/n): "

        if hp.save_every_steps is not None and hp.save_every_epochs is not None:
            print(
                "Warning: save_every_steps and save_every_epochs are both set. "
                "Only save by steps."
            )

        if hp.validate_every_epochs is None and hp.validate_every_steps is None:
            if input(tmpl.format("validate_every_*", "validation")) != "y":
                exit()

        if hp.save_every_epochs is None and hp.save_every_steps is None:
            if input(tmpl.format("save_every_*", "saving")) != "y":
                exit()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        time_str = time.strftime("%Y%m%dT%H%M%S")
        with open(self.run_dir / f"config-{time_str}.yml", "w") as f:
            yaml.dump(vars(self.hp), f)

        return self.training_loop()

    @command
    def validate(self, sanity: bool = False):
        scalar_dict = self.val_test_loop(
            f"Validating epoch {self.current_epoch} "
            + ("(sanity checking) " if sanity else "")
            + "...",
            self.sanity_check_dataloader if sanity else self.validation_dataloader,
            self.validation_step,
            self.Mode.VAL,
        )
        scalar_dict = flatten_dict(dict(val=scalar_dict))
        if scalar_dict and not sanity:
            scalar_dict["epoch"] = self.current_epoch
            self.logger.log(scalar_dict, self.global_step)
        return scalar_dict

    @command
    def test(self):
        scalar_dict = self.val_test_loop(
            f"Testing epoch {self.current_epoch} ...",
            self.testing_dataloader,
            self.testing_step,
            self.Mode.TEST,
        )
        scalar_dict = flatten_dict(dict(test=scalar_dict))
        if scalar_dict:
            scalar_dict["epoch"] = self.current_epoch
            self.logger.log(scalar_dict, self.global_step)
        return scalar_dict

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    @command
    def clear(self):
        hp = self.hp
        self.ls()
        if (hp.runs_root / self.name).exists():
            if input("Are you sure to clear? (y)\n").lower() == "y":
                self.try_rmtree(hp.runs_root / self.name)
            else:
                print(f"Not cleared.")
        else:
            print("Nothing to clear.")

    @command
    def ls(self):
        print_directory_tree(self.hp.runs_root / self.name)
