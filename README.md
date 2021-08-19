# TorchZQ: A PyTorch experiment runner built with [zouqi](https://github.com/enhuiz/zouqi)

## Installation

Install from PyPI:

```
pip install torchzq
```

Install the latest version:

```
pip install git+https://github.com/enhuiz/torchzq@main
```

## An Example for MNIST Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_model(self):
        return Net()

    def create_dataset(self):
        return datasets.MNIST(
            "../data",
            train=self.training,
            download=True,
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

    def training_step(self, batch, optimizer_index):
        x, y = self.prepare_batch(batch)
        loss = F.nll_loss(self.model(x), y)
        return loss, {"nll_loss": loss.item()}

    @torch.no_grad()
    def testing_step(self, batch, batch_index):
        x, y = self.prepare_batch(batch)
        y_ = self.model(x).argmax(dim=-1)
        return {"accuracy": (y_ == y).float().mean().item()}


if __name__ == "__main__":
    torchzq.start(Runner)
```

## Run an Example

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

## Supported Features

- [x] Model checkpoints
- [x] Logging (Weights & Biases)
- [x] Gradient accumulation
- [x] Configuration file
- [x] FP16
