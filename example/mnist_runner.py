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


class Runner(torchzq.Runner):
    def __init__(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--test-batch-size', default=1000)
        super().__init__(parser, 'mnist', save_every=1, epochs=10)

    def create_model(self):
        return Net()

    def create_data_loader(self):
        args = self.args
        if self.training:
            return torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=args.batch_size, shuffle=True)
        else:
            return torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=args.test_batch_size, shuffle=True)

    def prepare_batch(self, batch):
        x, y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        return x, y

    def predict(self, x):
        return x.argmax(dim=-1)

    def criterion(self, x, y):
        return F.nll_loss(x, y)

    def evaluate(self, x, y):
        x = torch.stack(x)
        y = torch.stack(y)
        correct = (x == y).sum()
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(y), 100. * correct / len(y)))


if __name__ == "__main__":
    Runner().run()
