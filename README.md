# TorchZQ: a PyTorch experiment runner

## Installation

Install from PyPI (latest):

```
pip install torchzq --pre --upgrade
```

## A customized runner for MNIST classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchzq


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
    class HParams(torchzq.Runner.HParams):
        lr: float = 1e-3

    hp: HParams

    def create_model(self):
        return Net()

    def create_dataloader(self, mode):
        hp = self.hp
        dataset = datasets.MNIST(
            "../data",
            train=mode == "training",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.nj,
            shuffle=mode == mode.TRAIN,
            drop_last=mode == mode.TRAIN,
        )

    def create_metrics(self):
        metrics = super().create_metrics()

        def early_stop(count):
            if count >= 2:
                # the metric does not go down for the latest two validations
                self.hp.max_epochs = -1  # this terminates the training

        metrics.add_metric("val/nll_loss", [early_stop])
        return metrics

    def prepare_batch(self, batch, _):
        x, y = batch
        x = x.to(self.hp.device)
        y = y.to(self.hp.device)
        return x, y

    def training_step(self, batch, optimizer_index):
        x, y = batch
        loss = F.nll_loss(self.model(x), y)
        return loss, {"nll_loss": loss.item()}

    @torch.no_grad()
    def testing_step(self, batch, batch_index):
        x, y = batch
        y_ = self.model(x).argmax(dim=-1)
        return {"accuracy": (y_ == y).float().mean().item()}


if __name__ == "__main__":
    Runner().start()
```

## Execute the runner

**Training**

```
tzq example/config/mnist.yml train
```

**Testing**

```
tzq example/config/mnist.yml test
```

**Weights & Biases**

Before you run, login [Weights & Biases](https://docs.wandb.ai/quickstart) first.

```
pip install wandb # install weight & bias client
wandb login       # login
```

## Supported features

- [x] Model checkpoints
- [x] Logging (Weights & Biases)
- [x] Gradient accumulation
- [x] Configuration file
- [x] FP16
