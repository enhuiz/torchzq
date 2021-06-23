import numpy as np
import torch
import torch.nn as nn
from typing import Callable

from .utils import ItemProperty


class _Metric(nn.Module):
    count = ItemProperty()
    minima = ItemProperty()

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)
        self.register_buffer("_count", torch.zeros([]).long())
        self.register_buffer("_minima", torch.full([], np.inf))

    def reset(self):
        self.count = 0
        self.minima = np.inf

    def forward(self, value):
        if value >= self.minima:
            self.count += 1
            for callback in self.callbacks:
                callback(self.count)
        else:
            self.count = 0
        self.minima = min(self.minima, value)

    def __str__(self):
        return f"Metric(count={self.count:.4g}, minima={self.minima:.4g})"

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return dict(count=self.count, minima=self.minima)


class Metrics(nn.ModuleDict):
    def __init__(self, metrics: dict[str, list[Callable]] = {}):
        super().__init__({n: _Metric(cs) for n, cs in metrics.items()})

    def forward(self, stat_dict):
        for name, metric in self.items():
            if name not in stat_dict:
                raise KeyError(
                    f'Metric "{name}" is not found in the stat_dict. '
                    f'Possible metrics are: {", ".join(stat_dict.keys())}.'
                )
            metric(stat_dict[name])

    def to_dict(self):
        return {f"{n}/{k}": v for n, m in self.items() for k, v in m.to_dict().items()}
