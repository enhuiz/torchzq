import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Callable


class _Metric(nn.Module):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)
        self.register_buffer("count", torch.zeros([]))
        self.register_buffer("minima", torch.full([], np.inf))

    def reset(self):
        self.count *= 0
        self.minima *= np.inf

    def forward(self, value):
        if value >= self.minima:
            self.count += 1
            for callback in self.callbacks:
                callback(self.count)
        else:
            self.count = torch.zeros_like(self.count)
        self.minima = torch.minimum(self.minima, torch.full_like(self.minima, value))

    def __str__(self):
        return f"Metric(count={self.count:.4g}, minima={self.minima:.4g})"

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return dict(count=self.count.item(), minima=self.minima.item())


class Metrics(nn.ModuleDict):
    def __init__(self, metrics: dict[str, list[Callable]] = {}):
        super().__init__({n: _Metric(cs) for n, cs in metrics.items()})

    def forward(self, stat_dict):
        for name, metric in self.items():
            if name not in stat_dict:
                raise KeyError(f"Metric {name} is not found in the stat_dict.")
            metric(stat_dict[name])

    def to_dataframe(self):
        return pd.DataFrame([{"name": n} | m.to_dict() for n, m in self.items()])
