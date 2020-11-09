import contextlib
import argparse
import numpy as np
import torch
import zouqi

from .base import BaseRunner


@contextlib.contextmanager
def autocast_if(cond):
    try:
        if cond:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    except:
        pass


class LegacyRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def feed(self, x):
        return self.model(x)

    def criterion(self, x, y):
        raise NotImplementedError

    def prepare_batch(self, batch):
        raise NotImplementedError

    def step(self, batch):
        args = self.args

        model = self.model
        logger = self.logger

        x, y = self.prepare_batch(batch)

        if self.training:
            optimizer = self.optimizer
            optimizer.set_lr(self.args.lr())

            with autocast_if(args.use_fp16):
                x = self.feed(x)
                loss = self.criterion(x, y)

            if args.use_fp16:
                self.scaler.scale(loss / args.update_every).backward()
            else:
                (loss / args.update_every).backward()

            if (model.iteration + 1) % args.update_every == 0:
                if args.use_fp16:
                    self.scaler.unscale_(optimizer)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                model.zero_grad()

            logger.add_scalar("lr", optimizer.get_lr())
        else:
            with torch.no_grad():
                x = self.feed(x)
                loss = self.criterion(x, y)

        logger.add_scalar("loss", loss.item())

    @zouqi.command(inherit=True)
    def train(self, *args, update_every: int = 1, **kwargs):
        self.update_args(dict(update_every=update_every))
        super().train(*args, **kwargs)
