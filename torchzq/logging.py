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
    def __init__(self, log_dir, smoothing=[], **kwargs):
        super().__init__(log_dir, **kwargs)
        self._smoothing = smoothing
        self._meters = defaultdict(EMAMeter)
        self._buffer = {}
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
                    curried = partial(_method, tag, value, **kwargs)
                    self._delayed_addings.append(curried)
                    if _name in ["add_scalar", "add_text"]:
                        self._buffer[tag] = value

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
        if type(value) is float:
            value = f"{value:.4g}"
        return value

    def _render_value(self, key):
        value = self._buffer[key]
        for pattern in self._smoothing:
            if re.match(pattern, key) is not None:
                value = self.meters[key](value)
        return self._prettify(value)

    def _perform_addings(self, global_step):
        for adding in self._delayed_addings:
            adding(global_step=global_step)
        self.flush()
        self._delayed_addings.clear()

    def render(self, global_step, priority=[]):
        self._perform_addings(global_step)
        keys = self._priortize(self._buffer.keys(), priority)
        lines = [f"{key}: {self._render_value(key)}" for key in keys]
        self._buffer.clear()
        return lines


if __name__ == "__main__":
    logger = Logger("./tensorboard_test", [])
    logger.add_scalar("hi", 1)
    logger.add_text("hello", "world")
    print(logger.render(0))
