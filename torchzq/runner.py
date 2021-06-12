import argparse
import tqdm
import time
import yaml
import sys
import os
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import signal
import wandb
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial, cached_property
from collections import defaultdict
from itertools import islice

import zouqi

from .typing import Flag, Scheduled, _Scheduled, Optional, Literal
from .saver import Saver
from .scheduler import Scheduler
from .utils import print_directory_tree
from .interrupt import graceful_interrupt_handler

Mode = Literal["training", "validation", "testing"]


class Runner:
    args: Optional[argparse.Namespace] = None

    def __init__(
        self,
        name: str = "Unnamed",
        batch_size: int = 32,
        nj: int = min(os.cpu_count(), 12),
        device: str = "cuda",
        strict_loading: bool = True,
        runs_root: Path = Path("runs"),
        use_fp16: bool = False,
        ckpt: Path = None,
        lr: Scheduled = "1e-3",
        from_scratch: Flag = False,
        seed: int = 0,
        wandb_project: str = "",
    ):
        self._cached_modes = []
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ##############
    # Read-onlys #
    ##############

    @property
    def name(self):
        return self.args.name

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
    def ckpt_dir(self):
        return self.run_dir / "ckpts"

    @property
    def autocast_if_use_fp16(self):
        return partial(torch.cuda.amp.autocast, enabled=self.args.use_fp16)

    @property
    def Optimizer(self):
        return torch.optim.Adam

    @property
    def DataLoader(self):
        return DataLoader

    ############
    # Settable #
    ############

    @property
    def batch(self):
        return getattr(self, "_batch", None)

    @batch.setter
    def batch(self, value):
        setattr(self, "_batch", self.prepare_batch(value))

    def prepare_batch(self, batch):
        return batch

    ########
    # Lazy #
    ########

    @cached_property
    def logger(self):
        args = self.args
        wandb.init(config=args, project=args.wandb_project, group=self.name)
        return wandb

    @cached_property
    def saver(self):
        return Saver(self.ckpt_dir, self.args.strict_loading)

    @cached_property
    def scheduler(self):
        scheduler = Scheduler()
        for k, v in list(vars(self.args).items()):
            if isinstance(v, _Scheduled):
                setattr(self.args, k, scheduler.schedule(v))
        return scheduler

    @cached_property
    def training_data_loader(self):
        return self.create_data_loader("training")

    @cached_property
    def validation_data_loader(self):
        return self.create_data_loader("validation")

    @cached_property
    def testing_data_loader(self):
        return self.create_data_loader("testing")

    @cached_property
    def sanity_check_data_loader(self):
        return list(islice(self.validation_data_loader, 3))

    @cached_property
    def model(self):
        args = self.args
        saver = self.saver
        scheduler = self.scheduler
        model = self.create_model()
        model.to(args.device)
        model.epoch = 0
        model.iteration = 0
        saver.load(self.ckpt, model=model)
        scheduler.step(epoch=model.epoch, iteration=model.iteration)
        return model

    @cached_property
    def optimizers(self):
        args = self.args
        optimizers = self.create_optimizers()
        if len(optimizers) == 0:
            raise ValueError("There should be at least 1 optimizer but get 0.")
        for optimizer in optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)
        self.saver.load(self.ckpt, optimizers=optimizers)
        return optimizers

    @cached_property
    def scaler(self):
        args = self.args
        if args.use_fp16:
            scaler = torch.cuda.amp.GradScaler()
            self.saver.load(self.ckpt, scaler=scaler)
        else:
            scaler = None
        return scaler

    ############
    # Creators #
    ############

    def create_dataset(self, mode: Mode):
        raise NotImplementedError

    def create_data_loader(self, mode: Mode):
        args = self.args
        dataset = self.create_dataset(mode)
        data_loader = self.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.nj,
            shuffle=mode == "training",
            drop_last=mode == "training",
        )
        print("Dataset size:", len(dataset))
        return data_loader

    def create_model(self):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def create_optimizers(self):
        self.scheduler
        args = self.args
        optimizer = self.Optimizer(params=self.model.parameters(), lr=1)
        args.lr.add_listener(lambda lr: self.set_lr(optimizer, lr))
        return [optimizer]

    #########
    # Steps #
    #########

    def training_step(self, batch, optimizer_idx: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> dict:
        stat_dict = {}
        for i in range(len(self.optimizers)):
            stat_dict.update(self.training_step(batch, i)[1])
        stat_dict = {f"val_{k}": v for k, v in stat_dict.items()}
        return stat_dict

    def testing_step(self, batch, batch_idx: int) -> dict:
        raise NotImplementedError

    def training_step_with_optimization(self, batch) -> dict:
        args = self.args
        model = self.model
        stat_dict = {}
        start_time = time.time()

        model.iteration += 1
        self.scheduler.step(epoch=model.epoch, iteration=model.iteration)

        for optimizer_idx, optimizer in enumerate(self.optimizers):
            with self.autocast_if_use_fp16():
                loss, stat_dict_i = self.training_step(batch, optimizer_idx)

            stat_dict.update(stat_dict_i)

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

    #########
    # Loops #
    #########

    def training_loop(self):
        args = self.args
        model = self.model.train()

        while model.epoch < args.max_epochs:
            model.epoch += 1
            desc = f"Train: {model.epoch}/{args.max_epochs}"
            pbar = tqdm.tqdm(self.training_data_loader, desc=desc)

            def interrupt_callback(signum):
                print("Trying to gracefully shutdown ...")
                self.saver.dump()
                if signum == signal.SIGQUIT:
                    self.saver.save(model, self.optimizers, self.scaler)
                sys.exit(0)

            with graceful_interrupt_handler(callback=interrupt_callback):
                for self.batch in pbar:
                    try:
                        stat_dict = self.training_step_with_optimization(self.batch)
                        if model.iteration % args.log_every == 0:
                            self.logger.log(stat_dict, model.iteration)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            pbar.set_description("OOM! Skip batch.")
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad  # free some memory
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                self.saver.buffer(model, self.optimizers, self.scaler)
                if model.epoch % args.save_every == 0:
                    self.saver.dump()
                if model.epoch % args.validate_every == 0:
                    self.validate()
                    model.train()

    def val_test_loop(self, desc, data_loader, step_fn):
        model = self.model.eval()
        pbar = tqdm.tqdm(data_loader, desc=desc)
        stats_dict = defaultdict(list)
        for index, self.batch in enumerate(pbar):
            for k, v in step_fn(self.batch, index).items():
                stats_dict[k].append(v)
        stat_dict = {k: np.mean(v) for k, v in stats_dict.items()}
        stat_dict["epoch"] = model.epoch
        self.logger.log(stat_dict, model.iteration)

    ############
    # Commands #
    ############

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

        self.val_test_loop(
            "Validation sanity checking ...",
            self.sanity_check_data_loader,
            self.validation_step,
        )
        self.training_loop()

    @zouqi.command
    def validate(self):
        self.val_test_loop(
            f"Validating epoch {self.model.epoch} ...",
            self.validation_data_loader,
            self.validation_step,
        )

    @zouqi.command
    def test(self):
        self.val_test_loop(
            f"Testing epoch {self.model.epoch} ...",
            self.testing_data_loader,
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
