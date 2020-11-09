import os
import tqdm
import time
import shutil
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn
import functools
import contextlib
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial

import zouqi

from torchzq.parsing import boolean, flag, choices
from torchzq.saver import Saver
from torchzq.logging import Logger
from torchzq.utils import Timer
from torchzq.scheduler import Scheduler
from torchzq.events import create_events


def parse_modes(cls):
    modes = []
    for base in cls.__bases__:
        modes += parse_modes(base)
    modes += vars(cls).get("modes", [])
    return list(set(modes))


@contextlib.contextmanager
def locked(self, key, value):
    occupied = hasattr(self, key)
    if not occupied:
        setattr(self, key, value)
    yield
    if not occupied:
        delattr(self, key)


class MetaRunner(type):
    """
    Runner can be running in different modes, e.g. train, validate, test etc.
    Different mode will leads to different function behavior.
    This meta class automatically parse modes from the mode-determined function call.
    The mode-determined function decides what would be the mode it runs on and locks
    the mode variable until it returns.
    """

    def __new__(mcls, name, bases, attrs):
        @property
        def mode_prop(self):
            try:
                return self._mode
            except:
                raise AttributeError("Mode is not set, call a mode function first.")

        attrs["mode"] = mode_prop

        cls = super().__new__(mcls, name, bases, attrs)

        for mode in parse_modes(cls):
            try:
                f = getattr(cls, mode)
            except:
                raise AttributeError(f'Mode "{mode}" not defined in "{cls.__name__}".')

            def wrap(f, mode=mode):
                @functools.wraps(f)
                def wrapped(self, *args, **kwargs):
                    with locked(self, "_mode", mode):
                        return f(self, *args, **kwargs)

                return wrapped

            setattr(cls, mode, wrap(f))

        return cls


class BaseRunner(metaclass=MetaRunner):
    # different modes that a command could be in
    modes = ["train", "validate"]

    saver = None
    scheduler = None
    model = None
    optimizer = None
    scaler = None

    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        ckpt_dir: Path = "ckpt",
        strict_loading: boolean = True,
        log_dir: Path = "logs",
        quiet: flag = False,
        split: str = None,
        use_fp16: boolean = False,
    ):
        self.update_args(locals(), "self")
        self.events = create_events(
            "epoch_started",
            "epoch_completed",
            "iteration_started",
            "iteration_completed",
        )

    @property
    def name(self):
        return self.args.name

    @property
    def training(self):
        return self.mode == "train"

    @property
    def split(self):
        return self.args.split or self.mode

    @property
    def Dataset(self):
        raise NotImplementedError

    @property
    def Optimizer(self):
        return torch.optim.Adam

    def update_args(self, payload, ignored=[]):
        if type(ignored) is str:
            ignored = [ignored]
        ignored += ["__class__"]
        for key in ignored:
            if key in payload:
                del payload[key]
        self.args = getattr(self, "args", argparse.Namespace())
        self.args = argparse.Namespace(**{**vars(self.args), **payload})

    def autofeed(self, f, override={}, mapping={}):
        """Priority: 1. override, 2. parsed args 3. parameters' default"""
        assert hasattr(self, "args")
        payload = vars(self.args)

        def m(key):
            return mapping[key] if key in mapping else key

        signature = inspect.signature(getattr(f, "__init__", f))
        params = [p.name for p in signature.parameters.values()]

        kwargs = {k: payload[m(k)] for k in params if m(k) in payload}
        kwargs.update(override)

        return f(**kwargs)

    def create_scheduler(self):
        scheduler = Scheduler()
        if self.mode == "train":
            self.args.lr = scheduler.schedule(self.args.lr)
        return scheduler

    def create_logger(self, epoch=None, label=None):
        """Create a logger to log_dir/name/mode/split/epoch/label,
        Args:
            label: a short name to differentiate different experiments
        Returns:
            logger
        """
        args = self.args
        parts = map(
            lambda s: s.lstrip("/"),
            [
                self.name,
                self.mode,
                self.split,
                "" if epoch is None else str(epoch),
                "default" if label is None else str(label),
            ],
        )
        run_log_dir = Path(args.log_dir, *parts)
        loss_smoothing = [r"(\S+_)?loss(\S+_)?"]
        postfix = run_log_dir.relative_to(Path(args.log_dir, self.name))
        logger = Logger(run_log_dir, loss_smoothing, postfix=postfix)
        return logger

    def create_pbar(self, iterable):
        pbar = tqdm.tqdm(iterable, dynamic_ncols=True, disable=self.args.quiet)

        close = pbar.close

        create_line = lambda: tqdm.tqdm(bar_format="╰{postfix}")

        if self.args.quiet:
            items = {}
            line = create_line()

            timer = Timer(10)

            def set_line(i, s):
                items[i] = s
                if timer.timeup():
                    line.set_postfix_str(", ".join(items.values()) + "\n")
                    timer.restart()

            def close_all():
                close()
                line.close()

        else:
            lines = defaultdict(create_line)

            def set_line(i, s):
                lines[i].set_postfix_str(s)

            def close_all():
                close()
                for line in lines.values():
                    line.close()

        pbar.set_line = set_line
        pbar.close = close_all

        return pbar

    def create_dataset(self):
        raise NotImplementedError

    def create_data_loader(self, **kwargs):
        args = self.args
        dataset = self.create_dataset()
        data_loader = self.autofeed(
            DataLoader,
            override=dict(
                dataset=dataset,
                **kwargs,
            ),
            mapping=dict(
                num_workers="nj",
            ),
        )
        print("Dataset size:", len(dataset))
        return data_loader

    def create_model(self):
        raise NotImplementedError

    def create_optimizer(self, model):
        optimizer = self.autofeed(
            self.Optimizer,
            dict(params=model.parameters(), lr=1),
        )

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.args.device)

        def get_lr():
            return optimizer.param_groups[0]["lr"]

        def set_lr(lr):
            for g in optimizer.param_groups:
                g["lr"] = lr

        optimizer.get_lr = get_lr
        optimizer.set_lr = set_lr

        return optimizer

    def create_saver(self):
        args = self.args
        self.saver = Saver(args.ckpt_dir / self.name)

        if self.training and not self.saver.is_empty and not args.continue_:
            print("Some checkpoints exist and continue not set, skip training.")
            exit()

        self.saver.load(
            self.model,
            self.optimizer,
            self.scaler,
            epoch=args.epoch,
            strict=args.strict_loading,
        )

    def initialize(self):
        args = self.args

        # scheduler
        self.scheduler = self.create_scheduler()

        # model
        self.model = self.create_model()
        self.model.to(args.device)

        # fp16
        if args.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # optimizer
        if self.training:
            self.optimizer = self.create_optimizer(self.model)

        # saver
        self.create_saver()

        # logger
        if self.training:
            self.data_loader = self.create_data_loader(shuffle=True)
            self.logger = self.create_logger()
        else:
            self.data_loader = self.create_data_loader(shuffle=False)
            self.logger = self.create_logger(self.model.epoch, args.label)

        # events
        if self.training:

            def save(epoch):
                if epoch % args.save_every == 0:
                    self.saver.save(self.model, self.optimizer, self.scaler)

            self.events.epoch_completed.append(save)

    def step(self, batch):
        raise NotImplementedError

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast, enabled=self.args.use_fp16)

    def train_loop(self):
        args = self.args
        model = self.model
        logger = self.logger

        while model.epoch < args.max_epochs:
            model.epoch += 1
            self.events.epoch_started(model.epoch)

            pbar = self.create_pbar(self.data_loader)
            pbar.set_description(f"Train: {model.epoch}/{args.max_epochs}")
            for batch in pbar:
                model.iteration += 1
                self.events.iteration_started(model.iteration)
                self.scheduler.step(model.epoch, model.iteration)
                self.step(batch)
                for i, line in enumerate(logger.render(model.iteration)):
                    pbar.set_line(i, line)
                self.events.iteration_completed(model.iteration)
            pbar.close()

            self.events.epoch_completed(model.epoch)

    @torch.no_grad()
    def validate_loop(self):
        args = self.args
        model = self.model.eval()
        logger = self.logger

        pbar = self.create_pbar(self.data_loader)
        pbar.set_description(f"Validate @{model.epoch}")
        for index, batch in enumerate(pbar):
            self.events.iteration_started(index)
            self.step(batch)
            for i, line in enumerate(logger.render(index)):
                pbar.set_line(i, line)
            self.events.iteration_completed(index)
        mean = lambda l: sum(l) / len(l)
        for key in logger:
            if "loss" in key:
                print(f"Average {key}: {logger.average(key):.4g}")

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    @zouqi.command
    def train(
        self,
        lr: str = "1e-3",
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        continue_: flag = False,
        epoch: int = None,
    ):
        self.update_args(locals(), "self")
        self.initialize()
        self.train_loop()

    @zouqi.command
    def validate(self, epoch: int = None, label: str = None):
        self.update_args(locals(), "self")
        self.initialize()
        self.validate_loop()

    @zouqi.command
    def clear(self):
        if input("Are you sure to clear? (y)\n").lower() == "y":
            self.try_rmtree(Path(self.args.ckpt_dir, self.name))
            self.try_rmtree(Path(self.args.log_dir, self.name))
        else:
            print(f"Not cleared.")
