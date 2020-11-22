import argparse
from collections import defaultdict


class Event(list):
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)


class Events(argparse.Namespace):
    def __init__(self, *names):
        super().__init__(**{name: Event() for name in names})
