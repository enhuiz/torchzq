import os
import numpy as np
import tqdm
import argparse
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

from pathlib import Path

from torchzq import checkpoint
from torchzq.logger import Logger
from torchzq.utils import message_box


class Runner():
    def __init__(self, parser=None, name='default', batch_size=128, epochs=100, lr=1e-3, save_every=5, update_every=1):
        """args passed will be used as defaults.
        """
        parser = parser or argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--name', type=str, default=name)
        parser.add_argument('--ckpt-dir', type=Path, default='ckpt')
        parser.add_argument('--lr', type=float, default=lr)
        parser.add_argument('--batch-size', type=int, default=batch_size)
        parser.add_argument('--epochs', type=int, default=epochs)
        parser.add_argument('--nj', type=int, default=os.cpu_count())
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--last-epoch', type=int, default=None)
        parser.add_argument('--save-every', type=int, default=save_every)
        parser.add_argument('--continue', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--update-every', type=int, default=update_every)
        args = parser.parse_args()

        args.continue_ = getattr(args, 'continue')
        delattr(args, 'continue')

        self.args = args

        msg = '\n'.join([f'{k}: {v}' for k, v in sorted(vars(args).items())])
        print(message_box('Arguments', msg))

        self.logger = Logger(self.name)
        if args.command == 'train':
            self.logger.enable_recording()

    @property
    def name(self):
        return self.args.name

    def create_model(self):
        raise NotImplementedError

    def create_and_prepare_model(self):
        args = self.args
        model = self.create_model()
        model.name = self.name
        if self.training:
            model = model.train()
        else:
            model = model.eval()
        model = checkpoint.prepare(model,
                                   args.ckpt_dir,
                                   args.continue_,
                                   args.last_epoch)
        model = model.to(args.device)
        return model

    def create_dataset(self):
        raise NotImplementedError

    def create_data_loader(self):
        args = self.args
        ds = self.create_dataset()
        dl = DataLoader(ds, shuffle=self.training,
                        num_workers=args.nj,
                        batch_size=args.batch_size,
                        collate_fn=getattr(self, 'collate_fn', None))
        print('Dataset:', len(ds))
        return dl

    def prepare_batch(self, batch):
        return batch

    def criterion(self, x, y):
        raise NotImplementedError

    def monitor(self, x, y):
        pass

    def predict(self, x):
        return x

    @staticmethod
    def evaluate(fake, real):
        raise NotImplementedError

    @property
    def training(self):
        return self.args.command == 'train'

    def run(self):
        eval(f'self.{self.args.command}()')

    @staticmethod
    def create_pline(*args, **kwargs):
        return tqdm.tqdm(*args, **kwargs, bar_format='╰{postfix}', dynamic_ncols=True)

    def train(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_and_prepare_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        erange = range(model.last_epoch + 1,
                       model.last_epoch + 1 + args.epochs)
        plines = defaultdict(self.create_pline)
        for epoch in erange:
            self.step = epoch * len(dl)
            pbar = tqdm.tqdm(dl, dynamic_ncols=True)
            for batch in pbar:
                x, y = self.prepare_batch(batch)

                x = model(x)
                loss = self.criterion(x, y)
                (loss / args.update_every).backward()

                if (self.step + 1) % args.update_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.logger.log('step', self.step)
                self.logger.log('loss', loss.item())

                pbar.set_description(f'Epoch: {epoch}/{erange.stop}')
                for i, item in enumerate(self.logger.render(['step', 'loss'])):
                    plines[i].set_postfix_str(item)

                self.monitor(x, y)
                self.step += 1

            if (epoch + 1) % args.save_every == 0:
                model.save(epoch)

    def test(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_and_prepare_model()

        pbar = tqdm.tqdm(dl)

        fake = []
        real = []
        for batch in pbar:
            x, y = self.prepare_batch(batch)
            real += list(y)
            with torch.no_grad():
                x = model(x)
                x = self.predict(x)
            fake += list(x)

        self.evaluate(fake, real)
