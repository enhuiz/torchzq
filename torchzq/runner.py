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


class Runner:
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
            sma_windows={r"(\S+_)?loss": 200},
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
        dl = self.autofeed(
            DataLoader,
            override=dict(
                dataset=self.create_dataset(),
                shuffle=self.training,
                collate_fn=getattr(self, "collate_fn", None),
                worker_init_fn=getattr(self, "worker_init_fn", None),
            ),
            mapping=dict(num_workers="nj"),
        )
        print("Dataset size:", len(dl.dataset))
        return dl

    def create_optimizer(self, model):
        params = [{"params": model.parameters(), "initial_lr": 1}]
        return self.autofeed(torch.optim.Adam, dict(params=params, lr=1))

    def create_scheduler(self, optimizer, lr_lambda, last_epoch):
        args = self.args
        if isinstance(lr_lambda, float):
            return LambdaLR(optimizer, lambda _: lr_lambda, last_epoch)
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def prepare_batch(self, batch):
        return batch

    def criterion(self, x, y):
        raise NotImplementedError

    def monitor(self, x, y):
        pass

    def feed(self, model, x):
        return model(x)

    def predict(self, x):
        return x

    def evaluate(self, fake, real):
        raise NotImplementedError

    def run(self):
        func = getattr(self, self.command, None)
        if func is None:
            print(f"Command '{self.command}' does not exist.")
        else:
            func()

    @staticmethod
    def create_pline(*args, **kwargs):
        return tqdm.tqdm(*args, **kwargs, bar_format="╰{postfix}", dynamic_ncols=True)

    def train(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_and_prepare_model()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer, args.lr, model.last_epoch)

        erange = range(model.last_epoch + 1, model.last_epoch + 1 + args.epochs)
        plines = defaultdict(self.create_pline)

        for self.epoch in erange:
            self.step = self.epoch * len(dl)
            pbar = tqdm.tqdm(dl, dynamic_ncols=True)

            for self.batch in pbar:
                x, y = self.prepare_batch(self.batch)

                x = self.feed(model, x)
                loss = self.criterion(x, y)
                (loss / args.update_every).backward()

                if (self.step + 1) % args.update_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.logger.log("step", self.step)
                self.logger.log("lr", scheduler.get_last_lr()[0])
                self.logger.log("loss", loss.item())

                pbar.set_description(f"Epoch: {self.epoch}/{erange.stop}")
                items = self.logger.render(["step", "lr", "loss"])
                for i, item in enumerate(items):
                    plines[i].set_postfix_str(item)

                self.monitor(x, y)
                self.step += 1

            print("\n" * (len(plines) - 1))

            scheduler.step()

            if (self.epoch + 1) % args.save_every == 0:
                model.save(self.epoch)

    def test(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_and_prepare_model()

        pbar = tqdm.tqdm(dl)

        fake, real = [], []
        plines = defaultdict(self.create_pline)
        for step, batch in enumerate(pbar):
            x, y = self.prepare_batch(batch)
            real += list(y)
            with torch.no_grad():
                x = self.feed(model, x)
                loss = self.criterion(x, y)
                fake += list(self.predict(x))
            self.logger.log("step", step)
            self.logger.log("loss", loss.item())
            items = self.logger.render(["step", "loss"])
            for i, item in enumerate(items):
                plines[i].set_postfix_str(item)

        print("\n" * (len(plines) - 1))

        print(f'Average loss: {np.mean(list(self.logger.column("loss"))):.3g}')

        self.evaluate(fake, real)

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
