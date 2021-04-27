import os
import shutil
import argparse
import inspect
import numbers
import torch
import torch.nn as nn
from deprecated import deprecated
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from collections import defaultdict
from natsort import natsorted

import zouqi

from torchzq.parsing import boolean, flag
from torchzq.saver import Saver
from torchzq.scheduler import Scheduler
from torchzq.pbar import ProgressBar


class Mode(str):
    pass


class BaseRunner:
    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        strict_loading: boolean = True,
        ckpt_root: Path = Path("ckpt"),
        logs_root: Path = Path("logs"),
        quiet: flag = False,
        use_fp16: boolean = False,
        epoch: int = None,
    ):
        self.update_args(locals(), "self")
        self.modes = []

    @property
    def mode(self):
        if len(self.modes) == 0:
            raise RuntimeError('No mode has been set, call "self.switch_mode()" first.')
        return self.modes[0]

    @property
    def data_loader(self):
        return self.mode.data_loader

    @property
    def logger(self):
        return self.mode.logger

    @property
    def scheduler(self):
        return self.mode.scheduler

    @property
    def dataset(self):
        return self.data_loader.dataset

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
    def DataLoader(self):
        return DataLoader

    def update_args(self, payload, ignored=[]):
        if type(ignored) is str:
            ignored = [ignored]
        ignored += ["__class__"]
        for key in ignored:
            if key in payload:
                del payload[key]
        self.args = getattr(self, "args", argparse.Namespace())
        for k, v in payload.items():
            setattr(self.args, k, v)

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    @staticmethod
    def get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

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
        data_loader = self.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=args.nj,
            **kwargs,
        )
        print("==> Dataset size:", len(dataset))
        return data_loader

    def create_model(self):
        raise NotImplementedError

    def create_optimizers(self):
        args = self.args
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1)
        args.lr.add_listeners(lambda lr: self.set_lr(optimizer, lr))
        return [optimizer]

    def prepare_saver(self):
        if not hasattr(self, "saver"):
            self.saver = Saver(self.ckpt_dir, self.args.strict_loading)

    def prepare_logger(self):
        if self.logger is None:
            self.mode.logger = SummaryWriter(self.logs_dir)

    def prepare_data_loader(self):
        # data_loader should be before the model
        if self.data_loader is None:
            self.mode.data_loader = self.create_data_loader(shuffle=self.training)

    def prepare_scheduler(self):
        # scheduler should be before the model
        self.mode.scheduler = self.create_scheduler()

    def prepare_model(self):
        args = self.args
        if not hasattr(self, "model"):
            self.model = self.create_model()
            self.model.to(args.device)
            self.model.epoch = 0
            self.model.iteration = 0
            if self.training and not self.saver.empty and not args.continue_:
                print('Checkpoints exist and "--continue" not set, exited.')
                exit()
            self.saver.load(model=self.model, cache=True, epoch=args.epoch)
            self.scheduler.step(epoch=self.model.epoch, iteration=self.model.iteration)

    def prepare_optimizer(self):
        args = self.args
        if not hasattr(self, "optimizers") and self.training:
            self.optimizers = self.create_optimizers()
            for optimizer in self.optimizers:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(args.device)
            self.saver.load(optimizers=self.optimizers, epoch=args.epoch)

    def prepare_scaler(self):
        args = self.args
        if not hasattr(self, "scaler") and args.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            self.saver.load(scaler=self.scaler, epoch=args.epoch)

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

    def training_step(self, batch, optimizer_index) -> tuple[dict, dict]:
        raise NotImplementedError

    def validation_step(self, batch) -> dict:
        raise NotImplementedError

    def testing_step(self, batch):
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

        self.validate(sanity_check=True)

        self.switch_mode("train")

        args = self.args
        model = self.model
        logger = self.logger

        self.events.started()
        while model.epoch < args.max_epochs:
            model.epoch += 1
            self.events.epoch_started(model.epoch)

            pbar = ProgressBar(
                self.data_loader,
                args.quiet,
                desc=f"Train: {model.epoch}/{args.max_epochs}",
            )

            for batch in pbar:
                model.iteration += 1
                self.events.iteration_started(model.iteration)
                self.scheduler.step(model.epoch, model.iteration)
                for optimizer_index, optimizer in enumerate(self.optimizers):
                    try:
                        loss, stats = self.training_step(batch, optimizer_index)

                        if args.use_fp16:
                            self.scaler.scale(loss / args.update_every).backward()
                        else:
                            (loss / args.update_every).backward()

                        if model.iteration % args.update_every == 0:
                            if args.use_fp16:
                                self.scaler.unscale_(optimizer)

                            stats["grad_norm"] = nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                args.grad_clip_thres or 1e9,
                            ).item()

                            if args.use_fp16:
                                self.scaler.step(optimizer)
                                self.scaler.update()
                            else:
                                optimizer.step()

                            optimizer.zero_grad()

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
                    for key, val in stats.items():
                        pbar.update_line(key, f"{key}: {val:.4g}")
                        logger.add_scalar(key, val, model.iteration)
                    self.logger.flush()

            pbar.close()

            if model.epoch % args.save_every == 0:
                self.saver.save(self.model, self.optimizers, self.scaler)

            if model.epoch % args.validate_every == 0:
                self.validate()
                self.switch_mode("train")

    @zouqi.command
    @torch.no_grad()
    def validate(self, sanity_check=False):
        self.update_args(locals(), "self")
        self.switch_mode("validate")

        args = self.args
        model = self.model

        if sanity_check:
            pbar = ProgressBar(
                [batch for batch, _ in zip(self.data_loader, range(3))],
                args.quiet,
                desc="Sanity check for validation ...",
            )
        else:
            pbar = ProgressBar(
                self.data_loader,
                args.quiet,
                desc=f"Validating epoch: {model.epoch}",
            )

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
        self.ls()
        if input("Are you sure to clear? (y)\n").lower() == "y":
            self.try_rmtree(Path(self.args.ckpt_root, self.name))
            self.try_rmtree(Path(self.args.logs_root, self.name))
        else:
            print(f"Not cleared.")

    @zouqi.command
    def ls(self):
        self.print_tree(self.args.logs_root / self.name)
        self.print_tree(self.args.ckpt_root / self.name)

    @classmethod
    def print_tree(cls, root: Path, prefix: str = ""):
        print(f"{prefix}{root.name if prefix else root}")
        if root.is_dir():
            base = prefix.replace("─", " ").replace("├", "│").replace("└", " ")
            paths = natsorted(root.iterdir())
            for i, path in enumerate(paths):
                if i < len(paths) - 1:
                    cls.print_tree(path, base + "├── ")
                else:
                    cls.print_tree(paths[-1], base + "└── ")
