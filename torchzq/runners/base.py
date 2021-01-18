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
from natsort import natsorted
from treelib import Tree

from torch.utils.tensorboard import SummaryWriter

import zouqi

from torchzq.parsing import boolean, flag, choices
from torchzq.saver import Saver
from torchzq.scheduler import Scheduler
from torchzq.events import Events
from torchzq.pbar import create_pbar


class Mode(str):
    logger = None
    events = None
    data_loader = None


class BaseRunner:
    model = None
    scheduler = None
    optimizer = None
    scaler = None
    saver = None

    modes = []

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
        epoch: int = None,
    ):
        self.update_args(locals(), "self")

    @property
    def mode(self):
        if len(self.modes) == 0:
            raise RuntimeError('No mode has been set, call "self.switch_mode()" first.')
        return self.modes[0]

    @property
    def data_loader(self):
        return self.mode.data_loader

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def logger(self):
        return self.mode.logger

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

    def create_dataset(self):
        raise NotImplementedError

    def create_data_loader(self, **kwargs):
        args = self.args
        dataset = self.create_dataset()
        data_loader = self.autofeed(
            DataLoader,
            override=dict(
                dataset=dataset,
                batch_size=self.batch_size,
                **kwargs,
            ),
            mapping=dict(
                num_workers="nj",
            ),
        )
        print("==> Dataset size:", len(dataset))
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

    def create_events(self):
        args = self.args
        names = [
            "iteration_started",
            "iteration_completed",
            "started",
            "completed",
        ]
        if self.training:
            names += [
                "epoch_started",
                "epoch_completed",
            ]
        events = Events(*names)

        if self.training:

            def save(epoch):
                if epoch % args.save_every == 0:
                    self.saver.save(self.model, self.optimizer, self.scaler)

            def validate(epoch):
                if epoch % args.validate_every == 0:
                    self.validate()
                    self.switch_mode("train")

            events.epoch_completed.append(save)
            events.epoch_completed.append(validate)

        return events

    def prepare_saver(self):
        if self.saver is None:
            self.saver = Saver(self.ckpt_dir, self.args.strict_loading)

    def prepare_events(self):
        if self.events is None:
            self.mode.events = self.create_events()

    def prepare_logger(self):
        if self.logger is None:
            self.mode.logger = SummaryWriter(self.logs_dir)

    def prepare_data_loader(self):
        # data_loader should be before the model
        if self.data_loader is None:
            self.mode.data_loader = self.create_data_loader(shuffle=self.training)

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
            if self.training and not self.saver.empty and not args.continue_:
                print('Checkpoints exist and "--continue" not set, exited.')
                exit()
            self.saver.load(model=self.model, cache=True, epoch=args.epoch)

    def prepare_optimizer(self):
        args = self.args
        if self.optimizer is None and self.training:
            self.optimizer = self.create_optimizer(self.model)
            self.saver.load(optimizer=self.optimizer, epoch=args.epoch)

    def prepare_scaler(self):
        args = self.args
        if self.scaler is None and args.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            self.saver.load(scaler=self.scaler, epoch=args.epoch)

    def prepare_all(self):
        pipeline = lambda *levels: [f() for level in levels for f in level]
        pipeline(
            [self.prepare_saver, self.prepare_scheduler],
            [self.prepare_data_loader],
            [self.prepare_model],
            [self.prepare_optimizer, self.prepare_scaler],
            [self.prepare_events, self.prepare_logger],
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

    def step(self, batch):
        raise NotImplementedError

    @zouqi.command
    def train(
        self,
        lr: str = "1e-3",
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        validate_every: int = None,
        continue_: flag = False,
    ):
        if validate_every is None:
            validate_every = save_every

        self.update_args(locals(), "self")
        self.switch_mode("train")

        args = self.args
        model = self.model
        logger = self.logger

        self.events.started()
        while model.epoch < args.max_epochs:
            model.epoch += 1
            self.events.epoch_started(model.epoch)
            pbar = create_pbar(self.data_loader, args.quiet)
            pbar.set_description(f"Train: {model.epoch}/{args.max_epochs}")
            for batch in pbar:
                model.iteration += 1
                self.events.iteration_started(model.iteration)
                self.scheduler.step(model.epoch, model.iteration)
                try:
                    stats = self.step(batch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        pbar.update_line(0, "OOM! Skip batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                pbar.update_line(0, f"iteration: {model.iteration}")
                for l, (key, val) in enumerate(stats.items(), 1):
                    pbar.update_line(key, f"{key}: {val:.4g}")
                    logger.add_scalar(key, val, model.iteration)
                self.logger.flush()
                self.events.iteration_completed(model.iteration)
            pbar.close()
            self.events.epoch_completed(model.epoch)
        self.events.completed()

    @zouqi.command
    @torch.no_grad()
    def validate(self):
        self.update_args(locals(), "self")
        self.switch_mode("validate")

        args = self.args
        logger = self.logger
        model = self.model

        pbar = create_pbar(self.data_loader, args.quiet)
        pbar.set_description(f"Validate epoch: {model.epoch}")

        stats_list = defaultdict(list)
        self.events.started()
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
        self.events.completed()

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

    @zouqi.command
    def ls(self):
        self.print_tree(self.args.logs_root / self.name)
        self.print_tree(self.args.ckpt_root / self.name)

    @staticmethod
    def print_tree(root):
        tree = Tree()
        first = True

        for dirpath, _, files in os.walk(root):
            dirpath = Path(dirpath)

            if first:
                parent = None
            else:
                parent = dirpath.parent

            tree.create_node(
                tag=f"{dirpath if first else dirpath.name}/",
                identifier=dirpath,
                parent=parent,
            )

            first = False

            for f in natsorted(files):
                filepath = dirpath / f
                tree.create_node(
                    tag=f"{filepath.name} ({time.ctime(filepath.lstat().st_mtime)})",
                    identifier=filepath,
                    parent=dirpath,
                )

        tree.show()
