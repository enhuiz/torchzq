import numpy as np
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torchvision.utils import save_image

from torchzq.runners.base import BaseRunner
from torchzq.parsing import union, optional, lambda_, ignore_future_arguments


class GANRunner(BaseRunner):
    def __init__(self, parser=None, **kwargs):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--g-lr", type=union(float, lambda_), default=1e-4)
        parser.add_argument("--d-lr", type=union(float, lambda_), default=1e-4)
        parser.add_argument("--vis-dir", type=Path, default="vis")
        parser.add_argument("--vis-every", type=int, default=100)
        parser.add_argument("--gp-weight", type=float, default=10)
        parser.add_argument("--plr-weight", type=optional(float), default=None)
        parser.add_argument("--plr-decay", type=optional(float), default=None)
        parser = ignore_future_arguments(parser, ["lr"])
        super().__init__(parser, **kwargs)

    def g_feed(self, G, c):
        """Sample from generator.
        """
        raise NotImplementedError

    def d_feed(self, D, x, c=None):
        """Feed the discriminator.
        """
        raise NotImplementedError

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

    def plr_loss(self, conditions, images):
        args = self.args

        if args.plr_weight is None:
            return torch.zeros([])

        num_pixels = images.shape[2] * images.shape[3]
        pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
        outputs = (images * pl_noise).sum()

        pl_grads = torch_grad(
            outputs=outputs,
            inputs=conditions,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        pl = (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

        if hasattr(self, "pl_ema"):
            plr_loss = args.plr_weight * ((pl - self.pl_ema) ** 2).sum()
        else:
            self.pl_ema = pl

        self.pl_ema = self.decay * self.pl_ema + (1 - args.plr_decay) * pl

        return plr_loss

    def train(self):
        args = self.args

        dl = self.create_data_loader()

        model = self.create_and_prepare_model()
        G, D = model

        g_optimizer = self.create_optimizer(G)
        d_optimizer = self.create_optimizer(D)

        g_scheduler = self.create_scheduler(g_optimizer, args.g_lr, model.last_epoch)
        d_scheduler = self.create_scheduler(d_optimizer, args.d_lr, model.last_epoch)

        erange = range(model.last_epoch + 1, model.last_epoch + 1 + args.epochs)
        plines = defaultdict(self.create_pline)

        for self.epoch in erange:
            self.step = self.epoch * len(dl)
            pbar = tqdm.tqdm(dl, dynamic_ncols=True)

            for self.batch in pbar:
                c, x_real = self.prepare_batch(self.batch)

                # train d
                d_optimizer.zero_grad()

                x_fake = self.g_feed(G, c)
                x_real.requires_grad_()
                d_fake_output = self.d_feed(D, x_fake, c)
                d_real_output = self.d_feed(D, x_real, c)
                d_fake_loss = F.relu(1 - d_fake_output).mean()
                d_real_loss = F.relu(1 + d_real_output).mean()
                d_gp_loss = self.gp_loss(x_real, d_real_output)
                d_loss = d_fake_loss + d_real_loss + d_gp_loss
                d_loss.backward()

                d_optimizer.step()

                self.logger.log("d_fake_loss", d_fake_loss.item())
                self.logger.log("d_real_loss", d_real_loss.item())
                self.logger.log("d_gp_loss", d_gp_loss.item())
                self.logger.log("d_loss", d_fake_loss.item())
                self.logger.log("d_lr", d_scheduler.get_last_lr()[0])

                # train g
                g_optimizer.zero_grad()

                x_fake = self.g_feed(G, c)
                g_fake_output = self.d_feed(D, x_fake, c)
                g_fake_loss = g_fake_output.mean()
                g_plr_loss = self.plr_loss(c, x_fake)
                g_loss = g_fake_loss + g_plr_loss
                g_loss.backward()

                g_optimizer.step()

                self.logger.log("g_fake_loss", g_fake_loss.item())
                self.logger.log("g_plr_loss", g_plr_loss.item())
                self.logger.log("g_loss", g_loss.item())
                self.logger.log("g_lr", g_scheduler.get_last_lr()[0])

                self.logger.log("step", self.step)
                pbar.set_description(f"Epoch: {self.epoch}/{erange.stop}")
                items = self.logger.render(["step"])
                for i, item in enumerate(items):
                    plines[i].set_postfix_str(item)

                self.monitor(x_fake, x_real)
                self.step += 1

            print("\n" * (len(plines) - 1))

            d_scheduler.step()
            g_scheduler.step()

            if (self.epoch + 1) % args.save_every == 0:
                model.save(self.epoch)

    def monitor(self, fake, real):
        args = self.args
        if self.step % args.vis_every == 0:
            path = Path(args.vis_dir, self.name, self.command, f"{self.step:06d}.gif")
            path.parent.mkdir(exist_ok=True, parents=True)
            nrow = min(args.batch_size, 4)
            save_image([*fake[:nrow], *real[:nrow]], path, nrow)

    def test(self):
        args = self.args

        dl = self.create_data_loader()

        model = self.create_and_prepare_model()
        G, D = model

        plines = defaultdict(self.create_pline)

        pbar = tqdm.tqdm(dl, dynamic_ncols=True)

        fakes = []
        reals = []

        for self.step, self.batch in enumerate(pbar):
            c, x_real = self.prepare_batch(self.batch)

            # test d
            with torch.no_grad():
                x_fake = self.g_feed(G, c)
            x_real.requires_grad_()
            with torch.no_grad():
                fake_output = self.d_feed(D, x_fake, c)
            real_output = self.d_feed(D, x_real, c)
            d_fake_loss = F.relu(1 - fake_output).mean()
            d_real_loss = F.relu(1 + real_output).mean()
            d_gp_loss = self.gp_loss(x_real, real_output)
            d_loss = d_fake_loss + d_real_loss + d_gp_loss

            self.logger.log("d_fake_loss", d_fake_loss.item())
            self.logger.log("d_real_loss", d_real_loss.item())
            self.logger.log("d_gp_loss", d_gp_loss.item())
            self.logger.log("d_loss", d_fake_loss.item())

            # test g
            with torch.no_grad():
                x_fake = self.g_feed(G, c)
                fake_output = self.d_feed(D, x_fake, c)
                g_fake_loss = fake_output.mean()
                g_plr_loss = self.plr_loss(c, x_fake)
                g_loss = g_fake_loss + g_plr_loss

                self.logger.log("g_fake_loss", g_fake_loss.item())
                self.logger.log("g_plr_loss", g_plr_loss.item())
                self.logger.log("g_loss", g_loss.item())

            self.logger.log("step", self.step)
            items = self.logger.render(["step"])
            for i, item in enumerate(items):
                plines[i].set_postfix_str(item)

            fakes += list(x_fake.cpu().detach())
            reals += list(x_real.cpu().detach())

        print("\n" * (len(plines) - 1))

        self.evaluate(fakes, reals)
