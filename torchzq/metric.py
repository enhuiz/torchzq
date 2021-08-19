import numpy as np
import torch
import torch.nn as nn
from typing import Literal

from .utils import ItemProperty


class Metric(nn.Module):
    count = ItemProperty()
    minima = ItemProperty()

    def __init__(self, callbacks, mode: Literal["min", "max"] = "min"):
        super().__init__()
        self.callbacks = list(callbacks)
        self.mode = mode
        self.register_buffer("_count", torch.zeros([]).long())
        self.register_buffer("_minima", torch.full([], np.inf))

    def reset(self):
        self.count = 0
        self.minima = np.inf

    @property
    def best_score(self):
        if self.mode == "min":
            return self.minima
        return -self.minima

    def forward(self, value=None):
        if value is not None:
            if self.mode == "max":
                value = -value
            if value >= self.minima:
                self.count += 1
            else:
                self.count = 0
            self.minima = min(self.minima, value)

        for callback in self.callbacks:
            callback(self.count)

    def to_dict(self):
        return dict(count=self.count, best_score=self.best_score)


class Metrics(nn.ModuleDict):
    def __init__(self, metric_dict: dict[str, Metric] = {}):
        super().__init__(metric_dict)

    def add_metric(self, name, callbacks, mode: Literal["min", "max"] = "min"):
        if name in self:
            raise ValueError(f'Metric "{name}" exists.')
        self[name] = Metric(callbacks, mode)

    def forward(self, stat_dict):
        for name, metric in self.items():
            if stat_dict is not None and name not in stat_dict:
                raise KeyError(
                    f'Metric "{name}" is not found in the stat_dict. '
                    f'Possible metrics are: {", ".join(stat_dict.keys())}.'
                )
            metric(stat_dict[name])

    def to_dict(self):
        return {f"{n}/{k}": v for n, m in self.items() for k, v in m.to_dict().items()}
