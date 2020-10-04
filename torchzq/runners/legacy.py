import argparse
import numpy as np
import torch
import operator
import zouqi

from .base import BaseRunner


class LegacyRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def feed(self, x):
        return self.model(x)

    def criterion(self, x, y):
        raise NotImplementedError

    def step(self, batch):
        args = self.args

        model = self.model
        logger = self.logger

        x, y = batch

        if self.training:
            optimizer = self.optimizer

            optimizer.set_lr(self.args.lr())

            x = self.feed(x)
            loss = self.criterion(x, y)

            if self.model.amp is None:
                (loss / args.update_every).backward()
            else:
                with model.amp.scale_loss(loss / args.update_every, optimizer) as loss:
                    loss.backward()

            if (model.iteration + 1) % args.update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            logger.add_scalar("lr", optimizer.get_lr())
        else:
            with torch.no_grad():
                x = self.feed(x)
                loss = self.criterion(x, y)

        logger.add_scalar("loss", loss.item())

    @zouqi.command
    def train(self, *args, update_every: int = 1, **kwargs):
        super().train(*args, **kwargs)
