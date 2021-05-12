import math
import time
import tqdm
import numbers
from collections import defaultdict

from .utils import EMAMeter


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterable, dynamic_ncols=True, mininterval=0.5, **kwargs):
        super().__init__(
            iterable,
            dynamic_ncols=dynamic_ncols,
            mininterval=mininterval,
            **kwargs,
        )
        self.mininterval = mininterval
        self.lines = defaultdict(self.create_line)
        self.emas = defaultdict(EMAMeter)

    def timeup(self):
        return time.time() - self.last_print_t > self.mininterval

    def create_line(self):
        return tqdm.tqdm(bar_format="╰ {unit}")

    def update_line(self, k, v, fmt=None):
        if isinstance(v, numbers.Number) and math.isfinite(v):
            v = self.emas[k](v)
            fmt = fmt or "{k}: {v:.4g}"
        else:
            fmt = fmt or "{k}: {v}"
        first_update = k not in self.lines
        line = self.lines[k]
        line.unit = fmt.format(k=k, v=v)
        if first_update:
            line.refresh()
        elif self.timeup():
            line.refresh()

    def close(self):
        super().close()
        for line in self.lines.values():
            line.close()
