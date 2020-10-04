#!/usr/bin/env python3
"""
An MNIST example for TorchZQ.
"""

import torchzq

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Runner(torchzq.LegacyRunner):
    def __init__(self):
        super().__init__()

    def create_model(self):
        return Net()

    def create_dataset(self, split):
        args = self.args
        if split == "train":
            return datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        else:
            return datasets.MNIST(
                "../data",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )

    def prepare_batch(self, batch):
        x, y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        return x, y

    def predict(self, x):
        return x.argmax(dim=-1)

    def criterion(self, x, y):
        return F.nll_loss(x, y)

    @torchzq.command
    def test(self, epoch=None, label: str = "default"):
        self.update_args(self.test.args)
        self.initialize()
        pbar = self.create_pbar(self.data_loader)
        fake, real = [], []
        for batch in pbar:
            x, y = self.prepare_batch(batch)
            fake += self.model(x).argmax(dim=-1).cpu().tolist()
            real += y.cpu().tolist()
        self.evaluate(fake, real)

    def evaluate(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        correct = (x == y).sum()
        print(
            "Test set: Accuracy: {}/{} ({:.0f}%)".format(
                correct, len(y), 100.0 * correct / len(y)
            )
        )


if __name__ == "__main__":
    Runner().run()
