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
import numbers
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

import zouqi

from torchzq.parsing import boolean, flag, choices
from torchzq.saver import Saver
from torchzq.scheduler import Scheduler
from torchzq.events import create_events
from torchzq.pbar import create_pbar


def parse_modes(cls):
    modes = []
    for base in cls.__bases__:
        modes += parse_modes(base)
    modes += vars(cls).get("modes", [])
    return list(set(modes))


@contextlib.contextmanager
def working_mode(self, new):
    old = getattr(self, "_mode", None)
    setattr(self, "_mode", new)
    yield
    if old is None:
        delattr(self, "_mode")
    else:
        setattr(self, "_mode", old)


class MetaRunner(type):
    """
    Runner can be running in different modes, e.g. train, validate, test etc.
    Different mode will leads to different function behavior.
    This meta class automatically parse modes from the mode-determined function call.
    The mode-determined function decides what would be the mode it runs on.
    """

    def __new__(mcls, name, bases, attrs):
        @property
        def mode(self):
            try:
                return self._mode
            except:
                raise AttributeError("Mode is not set, call a mode function first.")

        attrs["mode"] = mode

        cls = super().__new__(mcls, name, bases, attrs)

        for name in parse_modes(cls):
            try:
                f = getattr(cls, name)
            except:
                raise AttributeError(f'Mode "{name}" not defined in "{cls.__name__}".')

            def wrap(f, name=name):
                @functools.wraps(f)
                def wrapped(self, *args, **kwargs):
                    with working_mode(self, name):
                        return f(self, *args, **kwargs)

                return wrapped

            setattr(cls, name, wrap(f))

        return cls


class BaseRunner(metaclass=MetaRunner):
    modes = ["train", "validate"]

    # global states
    model = None
    scheduler = None
    optimizer = None
    scaler = None
    saver = None

    # mode states
    states = argparse.Namespace()

    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        strict_loading: boolean = True,
        ckpt_root: Path = "ckpt",
        logs_root: Path = "logs",
        quiet: flag = False,
        use_fp16: boolean = False,
    ):
        self.update_args(locals(), "self")

    @property
    def state(self):
        return getattr(self.states, self.mode)

    @property
    def data_loader(self):
        return self.state.data_loader

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def logger(self):
        return self.state.logger

    @property
    def name(self):
        return self.args.name

    @property
    def training(self):
        return self.mode == "train"

    @property
    def logs_dir(self):
        return self.args.logs_root / self.name / self.mode

    @property
    def ckpt_dir(self):
        return self.args.ckpt_root / self.name

    @property
    def events(self):
        return self.state.events

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast, enabled=self.args.use_fp16)

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

    def create_logger(self):
        return SummaryWriter(self.logs_dir)

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

    def init_global_state(self):
        args = self.args

        if self.model is not None:
            return

        # scheduler (should before the model)
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
        self.saver = Saver(self.ckpt_dir)

        if self.training and not self.saver.is_empty and not args.continue_:
            raise RuntimeError("Some checkpoints exist and continue not set.")

        self.saver.load(
            self.model,
            self.optimizer,
            self.scaler,
            epoch=args.epoch,
            strict=args.strict_loading,
        )

    def create_events(self):
        args = self.args

        names = [
            "iteration_started",
            "iteration_completed",
            "epoch_started",
            "epoch_completed",
        ]

        events = create_events(*names)

        if self.training:

            def save(epoch):
                if epoch % args.save_every == 0:
                    self.saver.save(self.model, self.optimizer, self.scaler)

            def validate(epoch):
                if epoch % args.validate_every == 0:
                    self.validate()
                    self.model.train()

            events.epoch_completed.append(save)
            events.epoch_completed.append(validate)

        return events

    def init_mode_state(self):
        args = self.args

        if hasattr(self.states, self.mode):
            return

        state = argparse.Namespace()

        # logger
        state.logger = self.create_logger()

        state.data_loader = self.create_data_loader(shuffle=self.training)

        state.events = self.create_events()

        setattr(self.states, self.mode, state)

    def init_state(self):
        self.init_global_state()
        self.init_mode_state()

    def step(self, batch):
        raise NotImplementedError

    @zouqi.command
    def train(
        self,
        lr: str = "1e-3",
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        validate_every: int = 1,
        continue_: flag = False,
        epoch: int = None,
    ):
        self.update_args(locals(), "self")
        self.init_state()

        args = self.args
        model = self.model
        logger = self.logger

        while model.epoch < args.max_epochs:
            model.epoch += 1
            self.events.epoch_started(model.epoch)
            pbar = create_pbar(self.data_loader, args.quiet)
            pbar.set_description(f"Train: {model.epoch}/{args.max_epochs}")
            for batch in pbar:
                model.iteration += 1
                self.events.iteration_started(model.iteration)
                self.scheduler.step(model.epoch, model.iteration)
                stats = self.step(batch)
                pbar.update_line(0, f"iteration: {model.iteration}")
                for l, (key, val) in enumerate(stats.items(), 1):
                    pbar.update_line(l, f"{key}: {val:.4g}")
                    logger.add_scalar(key, val, model.iteration)
                self.logger.flush()
                self.events.iteration_completed(model.iteration)
            pbar.close()
            self.events.epoch_completed(model.epoch)

    @zouqi.command
    @torch.no_grad()
    def validate(self, epoch: int = None):
        self.update_args(locals(), "self")
        self.init_state()

        args = self.args
        model = self.model.eval()
        logger = self.logger

        pbar = create_pbar(self.data_loader, args.quiet)
        pbar.set_description(f"Validate epoch: {model.epoch}")

        stats_list = defaultdict(list)
        self.events.epoch_started()
        for index, batch in enumerate(pbar):
            self.events.iteration_started(index)
            stats = self.step(batch)
            for l, (key, val) in enumerate(stats.items()):
                pbar.update_line(l, f"{key}: {val:.4g}")
                if isinstance(val, numbers.Number):
                    stats_list[key].append(val)
            self.events.iteration_completed(index)
        for key, val in stats_list.items():
            mean = sum(val) / len(val)
            self.logger.add_scalar(key, mean, model.epoch)
            print(f"Average {key}: {mean:.4g}.")
        self.logger.flush()
        self.events.epoch_completed()

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    @zouqi.command
    def clear(self):
        if input("Are you sure to clear? (y)\n").lower() == "y":
            self.try_rmtree(Path(self.args.ckpt_root, self.name))
            self.try_rmtree(Path(self.args.logs_root, self.name))
        else:
            print(f"Not cleared.")
