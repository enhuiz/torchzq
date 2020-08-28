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

from torchzq.runners.gan import GANRunner, nullcontext


class CGANRunner(GANRunner):
    def __init__(self, parser=None, **kwargs):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--plr-weight", type=float, default=0.01)
        parser.add_argument("--plr-decay", type=float, default=0.99)
        parser.add_argument("--plr-start", type=float, default=5000)
        parser.add_argument("--plr-every", type=float, default=32)
        super().__init__(parser, **kwargs)

    def plr_loss(self, styles, images):
        args = self.args

        if args.plr_weight == 0:
            return torch.zeros([])

        num_pixels = images.shape[2] * images.shape[3]
        pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
        outputs = (images * pl_noise).sum()

        pl_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=styles,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        pl = (pl_grads ** 2).sum(dim=1).mean(dim=0).sqrt()

        if hasattr(self, "pl_ema"):
            plr_loss = args.plr_weight * ((pl - self.pl_ema) ** 2).sum()
        else:
            plr_loss = torch.zeros([])
            self.pl_ema = pl

        decay = args.plr_decay
        with torch.no_grad():
            self.pl_ema = decay * self.pl_ema + (1 - decay) * pl

        return plr_loss

    def g_feed(self, batch, return_style=False):
        raise NotImplementedError

    def g_criterion(self, batch):
        args = self.args
        fake, style = self.g_feed(batch, return_style=True)
        with nullcontext() if self.training else torch.no_grad():
            fake_output = self.d_feed(fake, batch)

        if self.step >= args.plr_start and self.step % args.plr_every == 0:
            plr_loss = self.plr_loss(style, fake)
        else:
            plr_loss = torch.zeros([])

        return {
            "g_fake_loss": fake_output.mean(),
            "g_plr_loss": plr_loss,
        }
