import math
import numpy as np
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torchvision.utils import save_image

import zouqi
from torchzq.runners.base import BaseRunner
from torchzq.parsing import union, optional, lambda_
from torchzq.scheduler import Scheduler

try:
    from contextlib import nullcontext
except:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(*args, **kwargs):
        yield None


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--gp-weight", type=float, default=10)

    def gp_loss(self, images, outputs):
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

        grad = grad.view(n, -1)

        return self.args.gp_weight * ((grad.norm(2, dim=1) - 1) ** 2).mean()

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
        if self.command == "train":
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

        logger = self.logger

        real, label = batch

        G, D = self.model

        if self.training:
            g_optimizer, d_optimizer = self.optimizer
            g_optimizer.set_lr(args.g_lr())
            d_optimizer.set_lr(args.d_lr())

        # train d
        if self.training:
            d_optimizer.zero_grad()

        z = self.sample(len(real))
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
            logger.add_scalar(name, loss.item())
        logger.add_scalar("d_loss", d_loss.item())

        if self.training:
            d_loss.backward()
            d_optimizer.step()
            logger.add_scalar("d_lr", d_optimizer.get_lr())

        # train g
        if self.training:
            g_optimizer.zero_grad()

        z = self.sample(len(real))
        fake = self.feed(G, z, label)
        fake_output = self.feed(D, fake, label)
        g_loss = fake_output.mean()
        logger.add_scalar("g_loss", g_loss.item())

        if self.training:
            g_loss.backward()
            g_optimizer.step()
            logger.add_scalar("g_lr", g_optimizer.get_lr())

    @zouqi.command
    def train(self, *args, g_lr=1e-3, d_lr=1e-3, lr=None, **kwargs):
        self.args.g_lr = g_lr
        self.args.d_lr = d_lr
        super().train(*args, **kwargs)
