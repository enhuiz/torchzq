import argparse
import numpy as np
import torch
import operator
import zouqi

from .base import BaseRunner


class LegacyRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def feed(self, model, x):
        return model(x)

    def criterion(self, x, y):
        raise NotImplementedError

    def step(
        self,
        batch,
        model,
        logger,
        optimizer=None,
        scheduler=None,
    ):
        args = self.args

        x, y = batch

        if optimizer is not None:
            optimizer.set_lr(self.args.lr())

            x = self.feed(model, x)
            loss = self.criterion(x, y)

            if model.amp is not None:
                with model.amp.scale_loss(loss / args.update_every, optimizer) as loss:
                    loss.backward()
            else:
                (loss / args.update_every).backward()

            if (model.iteration + 1) % args.update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            logger.add_scalar("lr", optimizer.get_lr())
        else:
            with torch.no_grad():
                x = self.feed(model, x)
                loss = self.criterion(x, y)

        logger.add_scalar("loss", loss.item())

    @zouqi.command
    def train(self, *args, update_every: int = 1, **kwargs):
        super().train(*args, **kwargs)
