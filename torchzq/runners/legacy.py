import argparse
import numpy as np
import torch
import operator

from .base import BaseRunner


class LegacyRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def feed(self, model, x):
        return model(x)

    def criterion(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        return x

    def update(self, batch):
        args = self.args
        x, y = batch

        if self.training:
            x = self.feed(self.model, x)
            loss = self.criterion(x, y)

            if self.use_amp:
                with self.amp.scale_loss(
                    loss / args.update_every, self.optimizer
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                (loss / args.update_every).backward()

            if (self.step + 1) % args.update_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger.log("lr", self.scheduler.get_last_lr()[0])
        else:
            self.reals += list(y)
            with torch.no_grad():
                x = self.feed(self.model, x)
                loss = self.criterion(x, y)
                self.fakes += list(self.predict(x))

        self.logger.log("loss", loss.item())

    def test(self):
        self.reals = []
        self.fakes = []
        super().test()
        print(f'Average loss: {np.mean(list(self.logger.column("loss"))):.3g}')
        self.evaluate(self.fakes, self.reals)
