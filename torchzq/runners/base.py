import os
import tqdm
import shutil
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

from torchzq import checkpoint
from torchzq.parsing import union, lambda_, str2bool
from torchzq.logging import Logger, message_box


class BaseRunner(object):
    def __init__(
        self,
        parser=None,
        name="Unnamed",
        batch_size=128,
        epochs=100,
        lr=1e-3,
        save_every=5,
        update_every=1,
    ):
        """args passed will be used as defaults.
        """
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--name", type=str, default=name)
        parser.add_argument("--lr", type=union(float, lambda_), default=lr)
        parser.add_argument("--weight-decay", type=float, default=0)
        parser.add_argument("--batch-size", type=int, default=batch_size)
        parser.add_argument("--epochs", type=int, default=epochs)
        parser.add_argument("--nj", type=int, default=os.cpu_count())
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--last-epoch", type=int, default=None)
        parser.add_argument("--save-every", type=int, default=save_every)
        parser.add_argument("--continue", action="store_true")
        parser.add_argument("--update-every", type=int, default=update_every)
        parser.add_argument("--ckpt-dir", type=Path, default="ckpt")
        parser.add_argument("--log-dir", type=Path, default="logs")
        parser.add_argument("--recording", type=str2bool, default=True)
        args = parser.parse_args()

        args.continue_ = getattr(args, "continue")
        delattr(args, "continue")

        self.args = args

        lines = [f"{k}: {v}" for k, v in sorted(vars(args).items())]
        print(message_box("Arguments", "\n".join(lines)))

        self.logger = Logger(
            dir=Path(args.log_dir, self.name, self.command) if args.recording else None,
            sma_windows={r"(\S+_)?loss(\S+_)?": 200},
        )

    @property
    def name(self):
        return self.args.name

    @property
    def command(self):
        return self.args.command

    @property
    def training(self):
        return self.command == "train"

    def autofeed(self, callable, override={}, mapping={}):
        """Priority: 1. override, 2. parsed args 3. parameters' default
        """
        parameters = inspect.signature(callable).parameters

        def mapped(key):
            return mapping[key] if key in mapping else key

        def default(key):
            return parameters[key].default

        def getval(key):
            if key in override:
                return override[key]
            return getattr(self.args, mapped(key), default(key))

        return callable(**{key: getval(key) for key in parameters})

    def create_model(self):
        raise NotImplementedError

    def create_and_prepare_model(self):
        args = self.args
        model = self.create_model()

        if self.training:
            model = model.train()
        else:
            model = model.eval()

        model = checkpoint.prepare(
            model=model,
            ckpt_dir=Path(args.ckpt_dir, self.name),
            continue_=args.continue_,
            last_epoch=args.last_epoch,
        )

        model = model.to(args.device)

        return model

    def create_dataset(self):
        raise NotImplementedError

    def create_data_loader(self):
        args = self.args
        loader = self.autofeed(
            DataLoader,
            override=dict(
                dataset=self.create_dataset(),
                shuffle=self.training,
                collate_fn=getattr(self, "collate_fn", None),
                worker_init_fn=getattr(self, "worker_init_fn", None),
            ),
            mapping=dict(num_workers="nj"),
        )
        print("Dataset size:", len(loader.dataset))
        return loader

    def create_optimizer_impl(self, model):
        params = [{"params": model.parameters(), "initial_lr": 1}]
        return self.autofeed(torch.optim.Adam, dict(params=params, lr=1))

    def create_scheduler_impl(self, optimizer, lr, last_epoch):
        if isinstance(lr, float):
            return LambdaLR(optimizer, lambda _: lr, last_epoch)
        return LambdaLR(optimizer, lr, last_epoch)

    def create_optimizer(self):
        return self.create_optimizer_impl(self.model)

    def create_scheduler(self):
        return self.create_scheduler_impl(
            self.optimizer, self.args.lr, self.model.last_epoch
        )

    def run(self):
        func = getattr(self, self.command, None)
        if func is None:
            print(f"Command '{self.command}' does not exist.")
        else:
            func()

    def update(self, batch):
        raise NotImplementedError

    def prepare_train(self):
        self.loader = self.create_data_loader()
        self.model = self.create_and_prepare_model()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()

    def create_pbar(self):
        pbar = tqdm.tqdm(self.loader, dynamic_ncols=True)
        pbar.lines = defaultdict(lambda: tqdm.tqdm(bar_format="╰{postfix}"))
        close = pbar.close

        def close_all():
            close()
            for line in pbar.lines.values():
                line.close()

        pbar.close = close_all

        return pbar

    def prepare_batch(self, batch):
        raise NotImplementedError

    def train(self):
        args = self.args
        self.prepare_train()
        epoch_start = self.model.last_epoch + 1
        epoch_stop = self.model.last_epoch + 1 + args.epochs
        for self.epoch in range(epoch_start, epoch_stop):
            pbar = self.create_pbar()
            pbar.set_description(f"Epoch: {self.epoch}/{epoch_stop}")
            for self.step, batch in enumerate(pbar, self.epoch * len(self.loader)):
                self.logger.log("step", self.step)
                batch = self.prepare_batch(batch)
                self.update(batch)
                for i, item in enumerate(self.logger.render(["step"])):
                    pbar.lines[i].set_postfix_str(item)
            self.scheduler.step()
            if (self.epoch + 1) % self.args.save_every == 0:
                self.model.save(self.epoch)
        pbar.close()

    def prepare_test(self):
        self.loader = self.create_data_loader()
        self.model = self.create_and_prepare_model()

    def test(self):
        self.prepare_test()
        pbar = self.create_pbar()
        for self.step, batch in enumerate(pbar):
            self.logger.log("step", self.step)
            batch = self.prepare_batch(batch)
            self.update(batch)
            for i, item in enumerate(self.logger.render(["step"])):
                pbar.lines[i].set_postfix_str(item)
        pbar.close()

    @staticmethod
    def try_rmtree(path):
        if path.exists():
            shutil.rmtree(path)
            print(str(path), "removed.")

    def clear(self):
        if input("Are you sure to clear? (y)\n").lower() == "y":
            self.try_rmtree(Path(self.args.ckpt_dir, self.name))
            self.try_rmtree(Path(self.args.log_dir, self.name))
        else:
            print(f"Not cleared.")
