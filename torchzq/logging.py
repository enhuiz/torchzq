import re
import pandas as pd
import operator
import inspect
from pathlib import Path
from collections import defaultdict
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from .utils import EMAMeter


class Logger(SummaryWriter):
    def __init__(self, log_dir, smoothing=[], postfix="", **kwargs):
        """
        Args:
            *args, **kwargs
            postfix: postfix appended to the tag when added to tensorboard.
        """
        super().__init__(log_dir, **kwargs)
        postfix = re.sub("/+", "/", str(postfix).strip("/"))
        if postfix:
            postfix = "/" + postfix
        self._smoothing = smoothing
        self._postfix = postfix
        self._meters = defaultdict(EMAMeter)
        self._buffer = defaultdict(list)
        self._delayed_addings = []
        self._make_delayed_adders()

    def _make_delayed_adders(self):
        """
        Make the adder functions lazy and only to complete in the render function.
        If the adder is add_scalar or add_text, aslo save it to buffer for rendering.
        """
        for name, method in inspect.getmembers(self, inspect.ismethod):
            parameters = inspect.signature(method).parameters
            if name.startswith("add_") and "global_step" in parameters:

                def delayed(tag, value, _name=name, _method=method, **kwargs):
                    curried = partial(_method, tag + self._postfix, value, **kwargs)
                    self._delayed_addings.append(curried)
                    if _name in ["add_scalar", "add_text"]:
                        self._buffer[tag].append(value)

                setattr(self, name, delayed)

    def _priortize(self, l, priority):
        def helper(pair):
            i, s = pair
            p = len(l) - i
            for j, pattern in enumerate(priority):
                if re.match(pattern, s) is not None:
                    p = len(l) + len(priority) - j
            return "\0x00" * p + s

        return map(operator.itemgetter(1), sorted(enumerate(l), key=helper))

    @staticmethod
    def _prettify(value):
        try:
            if int(value) != value:
                value = f"{value:.4g}"
        except:
            pass
        return value

    def _render_value(self, tag):
        value = self._buffer[tag][-1]
        for pattern in self._smoothing:
            if re.match(pattern, tag) is not None:
                value = self._meters[tag](value)
        return self._prettify(value)

    def _perform_addings(self, global_step):
        for adding in self._delayed_addings:
            adding(global_step=global_step)
        self.flush()
        self._delayed_addings.clear()

    def render(self, global_step, priority=[]):
        self._perform_addings(global_step)
        self._buffer["iteration"].append(global_step)
        priority = ["iteration"] + priority
        tags = self._priortize(self._buffer, priority)
        lines = [f"{tag}: {self._render_value(tag)}" for tag in tags]
        return lines

    def keys(self):
        yield from self._buffer.keys()

    def values(self):
        yield from self._buffer.values()

    def items(self):
        yield from self._buffer.items()

    def average(self, tag):
        values = self._buffer[tag]
        return sum(values) / len(values)

    def __iter__(self):
        yield from self._buffer
