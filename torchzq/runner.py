import time
import yaml
import sys
import os
import shutil
import numbers
import numpy as np
import random
import torch
import torch.nn as nn
import signal
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from collections import defaultdict

import zouqi

from .typing import Flag, Scheduled, _Scheduled
from .saver import Saver
from .scheduler import Scheduler
from .pbar import ProgressBar
from .utils import print_directory_tree
from .interrupt import graceful_interrupt_handler


def immutable(name):
    def getter(self):
        return getattr(self, "__" + name, None)

    def setter(self, x):
        assert getter(self) == None, "Immutable variable cannot be set twice!"
        setattr(self, "__" + name, x)

    return property(getter, setter)


def delegate(delegate_cls, name):
    attrs = set(delegate_cls.__dict__.keys())

    def wrapped(cls):
        nonexist_attrs = attrs - set(cls.__dict__.keys())
        for attr in nonexist_attrs:
            getsrc = lambda self: getattr(self, name)
            getter = lambda self, attr=attr: getattr(getsrc(self), attr)
            setter = lambda self, x, attr=attr: setattr(getsrc(self), attr, x)
            setattr(cls, attr, property(getter, setter))
        return cls

    return wrapped


class Mode(str):
    logger = immutable("logger")
    data_loader = immutable("data_loader")


@delegate(Mode, "mode")
class Runner:
    model = immutable("model")
    saver = immutable("saver")
    optimizers = immutable("optimizers")
    scheduler = immutable("scheduler")
    scaler = immutable("scaler")
    args = immutable("args")

    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        strict_loading: bool = True,
        runs_root: Path = Path("runs"),
        quiet: Flag = False,
        use_fp16: bool = False,
        ckpt: Path = None,
        lr: Scheduled = "1e-3",
        from_scratch: Flag = False,
        seed: int = 0,
    ):
        self.modes = []
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def mode(self):
        if len(self.modes) == 0:
            raise RuntimeError('No mode has been set, call "self.switch_mode()" first.')
        return self.modes[0]

    @property
    def name(self):
        return self.args.name

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def training(self):
        return self.mode == "train"

    @property
    def ckpt(self):
        args = self.args
        if args.ckpt is not None:
            return args.ckpt
        if not args.from_scratch:
            return self.saver.latest_ckpt
        return None

    @property
    def run_dir(self):
        return self.args.runs_root / self.name

    @property
    def logs_dir(self):
        return self.run_dir / "logs" / self.mode

    @property
    def ckpt_dir(self):
        return self.run_dir / "ckpts"

    @property
    def events(self):
        return self.mode.events

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast, enabled=self.args.use_fp16)

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def Dataset(self):
        raise NotImplementedError

    @property
    def DataLoader(self):
        return DataLoader

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    @staticmethod
    def get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    def create_scheduler(self):
        scheduler = Scheduler()
        for k, v in list(vars(self.args).items()):
            if isinstance(v, _Scheduled):
                setattr(self.args, k, scheduler.schedule(v))
        return scheduler

    def create_dataset(self):
        raise NotImplementedError

    def create_data_loader(self, **kwargs):
        args = self.args
        dataset = self.create_dataset()
        data_loader = self.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=args.nj,
            **kwargs,
        )
        print("Dataset size:", len(dataset))
        return data_loader

    def create_model(self):
        raise NotImplementedError

    def create_optimizers(self):
        args = self.args
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1)
        args.lr.add_listeners(lambda lr: self.set_lr(optimizer, lr))
        return [optimizer]

    def prepare_saver(self):
        args = self.args
        if self.saver is None:
            self.saver = Saver(self.ckpt_dir, args.strict_loading)

    def prepare_logger(self):
        if self.logger is None:
            self.logger = SummaryWriter(self.logs_dir)

    def prepare_data_loader(self):
        # data_loader should be before the model
        if self.data_loader is None:
            self.data_loader = self.create_data_loader(
                shuffle=self.training, drop_last=self.training
            )

    def prepare_scheduler(self):
        # scheduler should be before the model
        if self.scheduler is None:
            self.scheduler = self.create_scheduler()

    def prepare_model(self):
        args = self.args
        if self.model is None:
            self.model = self.create_model()
            self.model.to(args.device)
            self.model.epoch = 0
            self.model.iteration = 0
            self.saver.load(self.ckpt, model=self.model)
            self.scheduler.step(epoch=self.model.epoch, iteration=self.model.iteration)

    def prepare_optimizer(self):
        args = self.args
        if self.optimizers is None:
            self.optimizers = self.create_optimizers()
            if len(self.optimizers) == 0:
                raise ValueError("There should be at least 1 optimizer but get 0.")
            for optimizer in self.optimizers:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(args.device)
            self.saver.load(self.ckpt, optimizers=self.optimizers)

    def prepare_scaler(self):
        args = self.args
        if self.scaler is None and args.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            self.saver.load(self.ckpt, scaler=self.scaler)

    @staticmethod
    def run_pipeline(*stages):
        for stage in stages:
            for func in stage:
                func()

    def prepare_all(self):
        self.run_pipeline(
            [self.prepare_saver, self.prepare_scheduler],
            [self.prepare_data_loader],
            [self.prepare_model],
            [self.prepare_optimizer, self.prepare_scaler],
            [self.prepare_logger],
        )

    def switch_mode(self, name):
        """
        Switch running mode.
        """
        # the first element in self.modes will be treated as the current mode
        if name in self.modes:
            i = self.modes.index(name)
            self.modes[0], self.modes[i] = self.modes[i], self.modes[0]
        else:
            self.modes.insert(0, Mode(name))
            self.prepare_all()
        if self.training:
            self.model.train()
        elif self.model is not None:
            self.model.eval()

    def training_step(self, batch, optimizer_idx: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> dict:
        stat_dict = {}
        for i in range(len(self.optimizers)):
            stat_dict.update(self.training_step(batch, i)[1])
        return stat_dict

    def testing_step(self, batch, batch_idx: int) -> dict:
        raise NotImplementedError

    def training_step_with_optimization(self, batch) -> dict:
        args = self.args
        model = self.model
        stat_dict = {}
        start_time = time.time()

        model.iteration += 1
        self.scheduler.step(model.epoch, model.iteration)

        for optimizer_idx, optimizer in enumerate(self.optimizers):
            with self.autocast_if_use_fp16():
                outputs = self.training_step(batch, optimizer_idx)

            if isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = outputs[0]
                stat_dict.update(outputs[1])

            if args.use_fp16:
                self.scaler.scale(loss / args.update_every).backward()
            else:
                (loss / args.update_every).backward()

            if model.iteration % args.update_every == 0:
                if args.use_fp16:
                    self.scaler.unscale_(optimizer)

                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    args.grad_clip_thres or 1e9,
                )

                stat_dict[f"grad_norm_{optimizer_idx}"] = grad_norm.item()

                if args.use_fp16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

        stat_dict["elapsed_time"] = time.time() - start_time

        return stat_dict

    def training_loop(self):
        if not self.training:
            print("Warning: You are calling training loop in non-training mode!")

        args = self.args
        model = self.model
        logger = self.logger

        while model.epoch < args.max_epochs:
            model.epoch += 1
            desc = f"Train: {model.epoch}/{args.max_epochs}"
            pbar = ProgressBar(self.data_loader, args.quiet, desc=desc)

            def interrupt_callback(signum):
                pbar.close()
                print("Trying to gracefully shutdown ...")
                self.saver.dump()
                if signum == signal.SIGQUIT:
                    self.saver.save(model, self.optimizers, self.scaler)
                sys.exit(0)

            with graceful_interrupt_handler(callback=interrupt_callback):
                for batch in pbar:
                    try:
                        stat_dict = self.training_step_with_optimization(batch)
                        pbar.update_line(0, f"iteration: {model.iteration}", "{v}")
                        for key, val in stat_dict.items():
                            pbar.update_line(key, val)
                        if model.iteration % args.log_every == 0:
                            for key, val in stat_dict.items():
                                logger.add_scalar(key, val, model.iteration)
                            logger.flush()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            pbar.update_line(0, "OOM! Skip batch.", "{v}")
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad  # free some memory
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                pbar.close()
                self.saver.buffer(model, self.optimizers, self.scaler)
                if model.epoch % args.save_every == 0:
                    self.saver.dump()
                if model.epoch % args.validate_every == 0:
                    self.validate()
                    self.switch_mode("train")

    def val_test_loop(self, desc, step, sanity_check=False):
        args = self.args
        model = self.model
        data_loader = self.data_loader
        if sanity_check:
            data_loader = [batch for batch, _ in zip(data_loader, range(3))]
        pbar = ProgressBar(data_loader, args.quiet, desc=desc)
        stats_dict = defaultdict(list)
        for index, batch in enumerate(pbar):
            stat_dict = step(batch, index)
            for key, val in (stat_dict or {}).items():
                pbar.update_line(key, val)
                if isinstance(val, numbers.Number):
                    stats_dict[key].append(val)
        for key, val in stats_dict.items():
            mean = sum(val) / len(val)
            self.logger.add_scalar(key, mean, model.epoch)
            print(f"Average {key}: {mean:.4g}.")
        self.logger.flush()

    @zouqi.command
    def train(
        self,
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        validate_every: int = None,
        update_every: int = 1,
        log_every: int = 10,
        grad_clip_thres: float = 1.0,
    ):
        args = self.args
        if validate_every is None:
            args.validate_every = save_every

        self.run_dir.mkdir(parents=True, exist_ok=True)
        time_str = time.strftime("%Y%m%dT%H%M%S")
        with open(self.run_dir / f"config-{time_str}.yml", "w") as f:
            yaml.dump(vars(args), f)

        self.switch_mode("validate")
        self.val_test_loop(
            "Validation loop sanity checking ...",
            self.validation_step,
            sanity_check=True,
        )
        self.switch_mode("train")
        self.training_loop()

    @zouqi.command
    def validate(self):
        self.switch_mode("validate")
        self.val_test_loop(
            f"Validating epoch {self.model.epoch} ...",
            self.validation_step,
        )

    @zouqi.command
    def test(self):
        self.switch_mode("test")
        self.val_test_loop(
            f"Testing epoch {self.model.epoch} ...",
            self.testing_step,
        )

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
