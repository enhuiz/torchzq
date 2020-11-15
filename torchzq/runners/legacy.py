import torch.nn as nn
import zouqi

from .base import BaseRunner


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

        with self.autocast_if_use_fp16():
            x = self.feed(x)
            loss = self.criterion(x, y)

        if self.training:
            optimizer = self.optimizer
            optimizer.set_lr(self.args.lr())

            if args.use_fp16:
                self.scaler.scale(loss / args.update_every).backward()
            else:
                (loss / args.update_every).backward()

            if (model.iteration + 1) % args.update_every == 0:
                if args.use_fp16:
                    self.scaler.unscale_(optimizer)

                if args.grad_clip_thres is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip_thres,
                    )
                    logger.add_scalar("grad_norm", grad_norm)

                if args.use_fp16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            logger.add_scalar("lr", optimizer.get_lr())

        logger.add_scalar("loss", loss.item())

    @zouqi.command
    def train(
        self,
        update_every: int = 1,
        grad_clip_thres: float = None,
        **kwargs,
    ):
        self.update_args(locals(), ["args", "kwargs", "self"])
        super().train(**kwargs)
