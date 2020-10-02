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


class BaseRunner(zouqi.Runner):
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
        args = self.parse_args()
        self.saver = Saver(args.ckpt_dir / self.name)

    @property
    def name(self):
        return self.args.name

    @property
    def Dataset(self):
        raise NotImplementedError

    @property
    def Optimizer(self):
        return torch.optim.Adam

    def create_logger(self, action, split, label=None):
        """Create a logger to {log_dir / name / action / split / (label)},
           keys will have the prefix {action/split}.
        Args:
            action: train/validate/test etc.
            split: data split
            label: a short name to differentiate different experiments
        Returns:
            logger
        """
        args = self.args
        parts = map(lambda s: s.lstrip("/"), [self.name, action, split, label or ""])
        log_dir = Path(args.log_dir, *parts)
        smoothing = [r"(\S+_)?loss(\S+_)?"]
        postfix = log_dir.relative_to(args.log_dir)
        logger = Logger(log_dir, smoothing, postfix=postfix)
        return logger

    def create_dataset(self, split=None):
        raise NotImplementedError

    def create_data_loader(self, split, shuffle=False):
        loader = self.autofeed(
            DataLoader,
            override=dict(
                dataset=self.create_dataset(split),
                shuffle=shuffle,
            ),
            mapping=dict(
                num_workers="nj",
            ),
        )
        print("Dataset size:", len(loader.dataset))
        return loader

    def create_model(self):
        raise NotImplementedError

    def prepare_amp(self, model, optimizer=None):
        if optimizer is None:
            args = [model]
        else:
            args = [model, optimizer]

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

        if len(args) == 1:
            return args[0]

        return tuple(args)

    def create_optimizer(self, model):
        optimizer = self.autofeed(self.Optimizer, dict(params=model.parameters(), lr=1))

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

    def create_pbar(self, loader):
        args = self.args

        pbar = tqdm.tqdm(loader, dynamic_ncols=True, disable=args.quiet)

        close = pbar.close

        create_line = lambda: tqdm.tqdm(bar_format="╰{postfix}")

        if args.quiet:
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

    def create_scheduler(self):
        scheduler = Scheduler()
        if self.command == "train":
            self.args.lr = scheduler.schedule(self.args.lr)
        return scheduler

    def prepare_batch(self, batch):
        raise NotImplementedError

    def step(self, batch, model, logger, optimizer=None):
        raise NotImplementedError

    @zouqi.command
    def train(
        self,
        lr: str = "Constant(1e-3)",
        weight_decay: float = 0,
        max_epochs: int = 100,
        save_every: int = 1,
        validate_every: int = None,
        continue_: flag = False,
        epoch: int = None,
    ):
        self.update_args(self.train.args)
        args = self.args

        scheduler = self.create_scheduler()

        model = self.create_model().to(args.device).train()
        optimizer = self.create_optimizer(model)
        model, optimizer = self.prepare_amp(model, optimizer)

        if not self.saver.empty and not continue_:
            print("Some checkpoints exist and continue not set, skip training.")
            exit()

        self.saver.load(model, optimizer, epoch=epoch, strict=args.strict_loading)

        split = args.split or "train"
        dl = self.create_data_loader(split, True)
        logger = self.create_logger("train", split)

        start_epoch = model.epoch

        while model.epoch < max_epochs:
            model.epoch += 1

            pbar = self.create_pbar(dl)
            pbar.set_description(f"Train: {model.epoch}/{max_epochs}")

            for batch in pbar:
                model.iteration += 1
                scheduler.step(model.epoch, model.iteration)
                batch = self.prepare_batch(batch)
                self.step(batch, model, logger, optimizer)
                for i, line in enumerate(logger.render(model.iteration)):
                    pbar.set_line(i, line)

            if model.epoch % save_every == 0:
                self.saver.save(model, optimizer)
            if validate_every and model.epoch % validate_every == 0:
                val_logger = self.validate(model=model)
                model.train()

            pbar.close()

    def prepare_test(self, action, split, epoch=None, model=None, label=None):
        """
        Args:
            action: logger action
            split: data split
            epoch: epoch to load, only when model is None
            model: loaded model
        Returns:
            model, pbar, logger
        """
        args = self.args

        if model is None:
            scheduler = self.create_scheduler()
            model = self.create_model().to(args.device)
            model = self.prepare_amp(model)
            self.saver.load(model, epoch=epoch, strict=args.strict_loading)
            scheduler.step(model.epoch, model.iteration)

        model.eval()

        dl = self.create_data_loader(split)
        pbar = self.create_pbar(dl)

        label = f"{label or ''}/{model.epoch}"
        logger = self.create_logger(action, split, label)

        return model, pbar, logger

    @zouqi.command
    @torch.no_grad()
    def validate(self, epoch: int = None, model: ignored = None):
        args = self.args

        model, pbar, logger = self.prepare_test(
            action="validate",
            split=args.split or "validate",
            epoch=epoch,
            model=model,
        )

        pbar.set_description(f"Validate @{model.epoch}")
        for index, batch in enumerate(pbar):
            batch = self.prepare_batch(batch)
            self.step(batch, model, logger)
            for i, line in enumerate(logger.render(index)):
                pbar.set_line(i, line)

        mean = lambda l: sum(l) / len(l)
        for key in logger:
            if "loss" in key:
                print(f"Average {key}: {logger.average(key):.4g}")

        return logger

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
