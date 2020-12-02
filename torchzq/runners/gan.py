import math
import numpy as np
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import zouqi
from torchzq.runners.base import BaseRunner
from torchzq.scheduler import Scheduler


class CombinedOptimizer(list):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def state_dict(self):
        state_dicts = []
        for optimizer in self:
            state_dicts.append(optimizer.state_dict())
        return state_dicts

    def load_state_dict(self, state_dicts):
        for i, optimizer in enumerate(self):
            optimizer.load_state_dict(state_dicts[i])


class GANRunner(BaseRunner):
    def __init__(
        self,
        *args,
        gp_weight: float = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.update_args(dict(gp_weight=gp_weight))

    def gp_loss(self, images, outputs):
        args = self.args

        if args.gp_weight <= 0:
            return 0

        n = images.shape[0]

        try:
            grad = torch.autograd.grad(
                outputs=outputs,
                inputs=images,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        except:
            # during validation, no_grad is set.
            return torch.zeros([])

        grad = grad.reshape(n, -1)

        return args.gp_weight * ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def create_optimizer(self, model):
        return CombinedOptimizer(
            [
                super().create_optimizer(model[0]),
                super().create_optimizer(model[1]),
            ]
        )

    def create_scheduler(self):
        args = self.args
        scheduler = Scheduler()
        if self.training:
            args.g_lr = scheduler.schedule(args.g_lr)
            args.d_lr = scheduler.schedule(args.d_lr)
        return scheduler

    @staticmethod
    def feed(model, z, label):
        try:
            return model(z, label)
        except TypeError:
            return model(z)

    def step(self, batch):
        args = self.args

        stats = {}

        logger = self.logger

        real, label = self.prepare_batch(batch)

        G, D = self.model

        if self.training:
            g_optimizer, d_optimizer = self.optimizer
            g_optimizer.set_lr(args.g_lr())
            d_optimizer.set_lr(args.d_lr())

        # train d
        if self.training:
            d_optimizer.zero_grad()

        z = self.sample(len(real))

        with self.autocast_if_use_fp16():
            fake = self.feed(G, z, label)
            real.requires_grad_()
            fake_output = self.feed(D, fake, label)
            real_output = self.feed(D, real, label)

            losses = {
                "d_fake_loss": F.relu(1 - fake_output).mean(),
                "d_real_loss": F.relu(1 + real_output).mean(),
                "d_gp_loss": self.gp_loss(real, real_output),
            }

            d_loss = 0
            for name, loss in losses.items():
                d_loss += loss
                stats[name] = loss.item()
            stats["d_loss"] = d_loss.item()

        if self.training:
            if args.use_fp16:
                self.scaler.scale(d_loss).backward()
                self.scaler.unscale_(d_optimizer)
            else:
                d_loss.backward()

            # grad clip here (TODO)

            if args.use_fp16:
                self.scaler.step(d_optimizer)
                self.scaler.update()
            else:
                d_optimizer.step()

            stats["d_lr"] = d_optimizer.get_lr()

        # train g
        if self.training:
            g_optimizer.zero_grad()

        z = self.sample(len(real))
        with self.autocast_if_use_fp16():
            fake = self.feed(G, z, label)
            fake_output = self.feed(D, fake, label)
            g_loss = fake_output.mean()
            stats["g_loss"] = g_loss.item()

        if self.training:
            if args.use_fp16:
                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(g_optimizer)
            else:
                g_loss.backward()

            # grad clip here (TODO)

            if args.use_fp16:
                self.scaler.step(g_optimizer)
                self.scaler.update()
            else:
                g_optimizer.step()

            stats["g_lr"] = g_optimizer.get_lr()

        return stats

    @zouqi.command(inherit=True)
    def train(self, *args, g_lr=1e-3, d_lr=1e-3, lr=None, **kwargs):
        self.args.g_lr = g_lr
        self.args.d_lr = d_lr
        super().train(*args, **kwargs)
