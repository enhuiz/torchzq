import os
import numpy as np
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import DataLoader

from pathlib import Path

from torchzq import checkpoint
from torchzq.logger import logger


class Runner():
    def __init__(self, parser=None, name='default', batch_size=128, epochs=100, lr=1e-3, save_every=5):
        """args passed will be used as defaults.
        """
        parser = parser or argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--name', type=str, default=name)
        parser.add_argument('--lr', type=float, default=lr)
        parser.add_argument('--batch-size', type=int, default=batch_size)
        parser.add_argument('--epochs', type=int, default=epochs)
        parser.add_argument('--nj', type=int, default=os.cpu_count())
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--last-epoch', type=int, default=None)
        parser.add_argument('--save-every', type=int, default=save_every)
        parser.add_argument('--continue', action='store_true')
        parser.add_argument('--test', action='store_true')
        args = parser.parse_args()

        args.continue_ = getattr(args, 'continue')
        delattr(args, 'continue')

        self.args = args
        print(args)

        self.logger = Logger(self.name)
        if args.command == 'train':
            self.logger.enable_recording()

    @property
    def name(self):
        return self.args.name

    def create_model(self):
        raise NotImplementedError

    def create_prepared_model(self):
        args = self.args
        model = self.create_model()
        if self.training:
            model = model.train()
        else:
            model = model.eval()
        model = checkpoint.prepare(model, args.continue_, args.last_epoch)
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
        raise NotImplementedError

    def criterion(self, x, y):
        raise NotImplementedError

    def monitor(self, x, y):
        pass

    @staticmethod
    def predict(x):
        raise NotImplementedError

    @staticmethod
    def evaluate(pd, gt):
        raise NotImplementedError

    @property
    def training(self):
        return self.args.command == 'train'

    def run(self):
        eval(f'self.{self.args.command}()')

    def train(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_prepared_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        ebar = tqdm.trange(model.last_epoch + 1,
                           model.last_epoch + 1 + args.epochs)
        for epoch in ebar:
            ebar.set_description(f'Epoch: {epoch}')
            self.iteration = epoch * len(dl)

            bbar = tqdm.tqdm(dl)
            for batch in bbar:
                x, y = self.prepare_batch(batch)

                x = model(x)
                loss = self.criterion(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.log('loss', loss.item())
                self.logger.log('iteration', self.iteration)

                desc = ', '.join(self.logger.render()).capitalize()
                bbar.set_description(desc)

                self.monitor(x, y)
                self.iteration += 1

            if (epoch + 1) % args.save_every == 0:
                model.save(epoch)

    def test(self):
        args = self.args

        dl = self.create_data_loader()
        model = self.create_prepared_model()

        pbar = tqdm.tqdm(dl)

        fake = []
        real = []
        for batch in pbar:
            x, y = self.prepare_batch(batch)
            real.append(y.cpu())
            with torch.no_grad():
                x = model(x)
            fake.append(self.predict(x).cpu())

        fake = torch.cat(fake, dim=0)
        real = torch.cat(real, dim=0)

        self.evaluate(fake, real)
