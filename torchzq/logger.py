import time
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict


class Timer():
    def __init__(self, interval):
        self.interval = interval
        self.restart()

    def restart(self):
        self.start_time = time.time()

    def timeup(self):
        return time.time() - self.start_time > self.interval


class Logger():
    def __init__(self, dir, flush_interval=10):
        self.dir = dir
        self.timer = Timer(flush_interval)
        self.entry = {}
        self.entries = []
        self.enable_recording()

    @property
    def path(self):
        return Path(self.dir, str(self.recording_start_time)).with_suffix('.csv')

    def enable_recording(self):
        self.recording_start_time = int(time.time())
        self.recording = True

    def disable_recording(self):
        self.recording = False

    def record(self):
        if self.entry:
            self.entry['timestamp'] = time.time()
            self.entries.append(self.entry)
            self.try_flush()

    def flush(self):
        if len(self.entries) > 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            df = self.to_frame()
            df.to_csv(self.path, index=None)

    def try_flush(self):
        if self.timer.timeup():
            self.flush()
            self.timer.restart()

    def to_frame(self):
        df = pd.DataFrame(self.entries)
        df = df.sort_values('timestamp')
        return df

    def log(self, k, v):
        self.entry[k] = self.to_basic(v)

    def clear(self):
        self.entry = {}

    @staticmethod
    def to_basic(v):
        if type(v) is torch.Tensor:
            try:
                v = v.item()
            except:
                v = v.tolist()
        return v

    @staticmethod
    def stringify(v):
        if type(v) is float:
            v = f'{v:.3g}'
        return v

    @staticmethod
    def priortize(priority):
        def priortizer(k):
            try:
                p = len(priority) - priority.index(k)
            except:
                p = 0
            return '\0x00' * p + k
        return priortizer

    def render(self, priority=[]):
        keys = sorted(self.entry.keys(), key=self.priortize(priority))
        items = [f'{k}: {self.stringify(self.entry[k])}' for k in keys]
        if self.recording:
            self.record()
        self.clear()
        return items

    def __del__(self):
        if self.recording:
            self.flush()
