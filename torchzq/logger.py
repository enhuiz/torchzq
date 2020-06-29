import time
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict


class Logger():
    def __init__(self, path, flush_interval=5):
        assert path.suffix == '.csv'

        self.path = path
        self.flush_interval = flush_interval
        self.last_flush_time = 0

        self.entry = {}
        self.records = []
        self.load()

        self.enable_recording()

    def load(self):
        try:
            df = pd.read_csv(self.path)
            self.records = df.to_dict('record')
        except:
            self.records = []

    def enable_recording(self):
        self.recording = True

    def disable_recording(self):
        self.recording = False

    def record(self):
        if self.entry:
            self.entry['timestamp'] = time.time()
            self.records.append(self.entry)
            self.try_flush()

    def flush(self):
        if len(self.records) > 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            df = self.data_frame
            df.to_csv(self.path, index=None)

    def try_flush(self):
        if time.time() - self.last_flush_time > self.flush_interval:
            self.flush()
            self.last_flush_time = time.time()

    @property
    def data_frame(self):
        df = pd.DataFrame(self.records)
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
