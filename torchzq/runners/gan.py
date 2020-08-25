import math
import numpy as np
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from torchvision.utils import save_image

from torchzq.runners.base import BaseRunner
from torchzq.parsing import union, optional, lambda_, ignore_future_arguments


class CombinedScheduler(list):
    def __init__(self, schedulers):
        super().__init__(schedulers)

    def step(self):
        for scheduler in self:
            scheduler.step()


class GANRunner(BaseRunner):
    def __init__(self, parser=None, **kwargs):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--g-lr", type=union(float, lambda_), default=1e-4)
        parser.add_argument("--d-lr", type=union(float, lambda_), default=1e-4)
        parser.add_argument("--vis-dir", type=Path, default="vis")
        parser.add_argument("--vis-every", type=int, default=100)
        parser.add_argument("--gp-weight", type=float, default=10)
        parser = ignore_future_arguments(parser, ["lr"])
        super().__init__(parser, **kwargs)

    def gp_loss(self, images, outputs):
        n = images.shape[0]

        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=images,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad = grad.view(n, -1)

        return self.args.gp_weight * ((grad.norm(2, dim=1) - 1) ** 2).mean()

    @property
    def G(self):
        return self.model[0]

    @property
    def D(self):
        return self.model[1]

    def g_feed(self, batch):
        raise NotImplementedError

    def d_feed(self, x, batch):
        raise NotImplementedError

    def get_real(self, batch):
        raise NotImplementedError

    def d_criterion(self, batch):
        with nullcontext() if self.training else torch.no_grad():
            fake = self.g_feed(batch)
            fake_output = self.d_feed(fake, batch)

        real = self.get_real(batch)
        real.requires_grad_()
        real_output = self.d_feed(real, batch)

        return {
            "d_fake_loss": F.relu(1 - fake_output).mean(),
            "d_real_loss": F.relu(1 + real_output).mean(),
            "d_gp_loss": self.gp_loss(real, real_output),
        }

    def g_criterion(self, batch):
        with nullcontext() if self.training else torch.no_grad():
            fake = self.g_feed(batch)
            fake_output = self.d_feed(fake, batch)

        return {
            "g_fake_loss": fake_output.mean(),
        }

    def create_optimizer(self):
        return (
            super().create_optimizer_impl(self.G),
            super().create_optimizer_impl(self.D),
        )

    def create_scheduler(self):
        args = self.args
        g_optimizer, d_optimizer = self.optimizer
        g_scheduler = super().create_scheduler_impl(
            g_optimizer, args.g_lr, self.model.last_epoch
        )
        d_scheduler = super().create_scheduler_impl(
            d_optimizer, args.d_lr, self.model.last_epoch
        )
        scheduler = CombinedScheduler([g_scheduler, d_scheduler])
        return scheduler

    def update(self, batch):
        args = self.args

        if self.training:
            g_optimizer, d_optimizer = self.optimizer
            g_scheduler, d_scheduler = self.scheduler

        # train d
        if self.training:
            d_optimizer.zero_grad()

        d_loss = 0
        for key, value in self.d_criterion(batch).items():
            self.logger.log(key, value.item())
            d_loss += value
        self.logger.log("d_loss", d_loss.item())

        if self.training:
            d_loss.backward()
            d_optimizer.step()
            self.logger.log("d_lr", d_scheduler.get_last_lr()[0])

        # train g
        if self.training:
            g_optimizer.zero_grad()

        g_loss = 0
        for key, value in self.g_criterion(batch).items():
            self.logger.log(key, value.item())
            g_loss += value
        self.logger.log("g_loss", g_loss.item())

        if self.training:
            g_loss.backward()
            g_optimizer.step()
            self.logger.log("g_lr", g_scheduler.get_last_lr()[0])
