import re
import time
import torch
import pandas as pd
import operator
from pathlib import Path
from collections import defaultdict
from functools import wraps
from itertools import islice


def message_box(title, content, width=None, aligner="<"):
    width = width or max(map(len, [title, *content.splitlines()])) + 8

    nb = width - 2  # number of blanks
    border = f"│{{: ^{nb}}}│"

    out = []
    out.append("┌" + "─" * nb + "┐")
    out.append(border.format(title.capitalize()))
    out.append("├" + "─" * nb + "┤")

    for line in content.splitlines():
        out.append(border.replace("^", aligner).format(line.strip()))

    out.append("└" + "─" * nb + "┘")

    return "\n".join(out)


class Timer:
    def __init__(self, interval):
        self.interval = interval
        self.restart()

    def restart(self):
        self.start_time = time.time()

    def timeup(self):
        return time.time() - self.start_time > self.interval


def throttle(func, seconds):
    timer = Timer(seconds)

    @wraps(func)
    def wrapped(*args, **kwargs):
        if timer.timeup():
            result = func(*args, **kwargs)
            timer.restart()
            return result

    return wrapped


class Logger:
    def __init__(self, dir, dump_interval=10, sma_windows={}):
        self.path = None if dir is None else self.create_path(dir)
        self.throttled_dump = throttle(self.dump, dump_interval)
        self.sma_windows = sma_windows
        self.records = [{}]

    @staticmethod
    def create_path(dir):
        return Path(dir, str(int(time.time()))).with_suffix(".csv")

    @staticmethod
    def convert_to_basic_type(value):
        if type(value) is torch.Tensor:
            try:
                value = value.item()
            except:
                value = value.tolist()
        return value

    @staticmethod
    def prettify(value):
        if type(value) is float:
            value = f"{value:.4g}"
        return value

    def priortize(self, l, priority):
        def helper(pair):
            i, s = pair
            p = len(l) - i
            for j, pattern in enumerate(priority):
                if re.match(pattern, s) is not None:
                    p = len(l) + len(priority) - j
            return "\0x00" * p + s

        return map(operator.itemgetter(1), sorted(enumerate(l), key=helper))

    @property
    def record(self):
        return self.records[-1]

    def log(self, key, value):
        self.record[key] = self.convert_to_basic_type(value)

    def column(self, key, reverse=False):
        records = reversed(self.records) if reverse else self.records
        for record in records:
            if key in record:
                yield record[key]

    def sma_window(self, key):
        for pattern in self.sma_windows:
            if re.match(pattern, key) is not None:
                return self.sma_windows[pattern]
        return 1

    def render(self, priority=[]):
        mean = lambda l: sum(l) / len(l) if len(l) > 1 else l[0]
        getval = lambda key: mean(
            list(islice(self.column(key, reverse=True), self.sma_window(key)))
        )

        ordered_keys = self.priortize(self.record.keys(), priority)
        items = [f"{key}: {self.prettify(getval(key))}" for key in ordered_keys]

        self.log("timestamp", time.time())
        self.throttled_dump()
        self.records.append({})

        return items

    def dump(self, path=None):
        path = path or self.path
        records = [record for record in self.records if "timestamp" in record]
        if path is not None and records:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            records = sorted(records, key=operator.itemgetter("timestamp"))
            df = pd.DataFrame(records)
            df.to_csv(path, index=None)

    def __del__(self):
        self.dump()
