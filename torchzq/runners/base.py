import os
import tqdm
import time
import shutil
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path

import zouqi
from torchzq.parsing import union, lambda_, str2bool, ignored, flag
from torchzq.saver import Saver
from torchzq.logging import Logger
from torchzq.utils import Timer
from torchzq.scheduler import Scheduler


class Events:
    class Event(list):
        def __call__(self, *args, **kwargs):
            for f in self:
                f(*args, **kwargs)

    def __init__(self):
        self._events = defaultdict(Events.Event)

    @property
    def epoch_started(self):
        return self._events["epoch_started"]

    @property
    def epoch_completed(self):
        return self._events["epoch_completed"]

    @property
    def iteration_started(self):
        return self._events["iteration_started"]

    @property
    def iteration_completed(self):
        return self._events["iteration_completed"]


class BaseRunner(zouqi.Runner):
    saver = None
    scheduler = None
    model = None
    optimizer = None

    def __init__(self, **kwargs):
        super().__init__(verbose=True, **kwargs)
        self.add_argument("--name", type=str, default="Unnamed")
        self.add_argument("--batch-size", type=int, default=32)
        self.add_argument("--nj", type=int, default=min(os.cpu_count(), 12))
        self.add_argument("--device", default="cuda")
        self.add_argument("--ckpt-dir", type=Path, default="ckpt")
        self.add_argument("--log-dir", type=Path, default="logs")
        self.add_argument("--strict-loading", type=str2bool, default=True)
        self.add_argument("--quiet", action="store_true")
        self.add_argument("--amp-level", choices=["O0", "O1", "O2", "O3"])
        self.add_argument("--split", type=str, default=None)
        self.events = Events()

    @property
    def name(self):
        return self.args.name

    @property
    def training(self):
        # do one thing at one time
        return self.command == "train"

    @property
    def split(self):
        return self.args.split or self.args.command

    @property
    def Dataset(self):
        raise NotImplementedError

    @property
    def Optimizer(self):
        return torch.optim.Adam

    def create_scheduler(self):
        scheduler = Scheduler()
        if self.command == "train":
            self.args.lr = scheduler.schedule(self.args.lr)
        return scheduler

    def create_logger(self, label=None, epoch=None):
        """Create a logger to {log_dir / name / command / split / (label)},
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
                self.command,
                self.split,
                "default" if label is None else str(label),
                "" if epoch is None else str(epoch),
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

    def initialize_amp(self):
        if self.optimizer is None:
            args = [self.model]
        else:
            args = [self.model, self.optimizer]

        try:
            from apex import amp
        except:
            amp = None

        amp_level = self.args.amp_level
        if amp_level is not None and amp is not None:
            args = list(amp.initialize(*args, opt_level=amp_level))
        else:
            amp = None

        args[0].amp = amp

        self.model = args[0]

        if len(args) > 1:
            self.optimizer = args[1]

    def initialize_saver(self):
        args = self.args
        self.saver = Saver(args.ckpt_dir / self.name)
        if self.training and not self.saver.is_empty and not args.continue_:
            print("Some checkpoints exist and continue not set, skip training.")
            exit()
        self.saver.load(
            self.model,
            self.optimizer,
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
        self.model.amp = None

        # optimizer
        if self.training:
            self.optimizer = self.create_optimizer(self.model)

        # amp
        self.initialize_amp()

        # saver
        self.initialize_saver()

        # logger
        if self.training:
            self.data_loader = self.create_data_loader(shuffle=True)
            self.logger = self.create_logger()
        else:
            self.data_loader = self.create_data_loader(shuffle=False)
            self.logger = self.create_logger(args.label, self.model.epoch)

        # events
        if self.training:

            def save(epoch):
                if epoch % args.save_every == 0:
                    self.saver.save(self.model, self.optimizer)

            self.events.epoch_completed.append(save)

    def prepare_batch(self, batch):
        raise NotImplementedError

    def step(self, batch):
        raise NotImplementedError

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
                batch = self.prepare_batch(batch)
                self.step(batch)
                for i, line in enumerate(logger.render(model.iteration)):
                    pbar.set_line(i, line)
                self.events.iteration_completed(model.iteration)
            pbar.close()

            self.events.epoch_completed(model.epoch)

    @zouqi.command
    def train(
        self,
        lr: str = "Constant(1e-3)",
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        continue_: flag = False,
        epoch: int = None,
    ):
        self.initialize()
        self.train_loop()

    @torch.no_grad()
    def validate_loop(self):
        args = self.args
        model = self.model.eval()
        logger = self.logger

        pbar = self.create_pbar(self.data_loader)
        pbar.set_description(f"Validate @{model.epoch}")
        for index, batch in enumerate(pbar):
            self.events.iteration_started(index)
            batch = self.prepare_batch(batch)
            self.step(batch)
            for i, line in enumerate(logger.render(index)):
                pbar.set_line(i, line)
            self.events.iteration_completed(index)
        mean = lambda l: sum(l) / len(l)
        for key in logger:
            if "loss" in key:
                print(f"Average {key}: {logger.average(key):.4g}")

    @zouqi.command
    def validate(self, epoch: int = None, label: str = None):
        self.initialize()
        self.validate_loop()

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    @zouqi.command
    def clear(self):
        if input("Are you sure to clear? (y)\n").lower() == "y":
            self.try_rmtree(Path(self.args.ckpt_dir, self.name))
            self.try_rmtree(Path(self.args.log_dir, self.name))
        else:
            print(f"Not cleared.")
