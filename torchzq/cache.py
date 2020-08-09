import pickle
import inspect
from pathlib import Path
from urllib.parse import urlencode
from collections import defaultdict


def dumbrw(*args, **kwargs):
    pass


def pickle_dumper(obj, path):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def pickle_loader(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Cache:
    def __init__(self, folder, verbose=False):
        self.folder = Path(folder)
        self.buffer = defaultdict(dict)
        self.verbose = verbose

    def load(self, loader, name, key):
        obj = None
        if name in self.buffer and key in self.buffer[name]:
            obj = self.buffer[name][key]
        else:
            path = Path(self.folder, name, key)
            if path.exists():
                obj = loader(path)
            self.buffer[name][key] = obj
        if self.verbose and obj is not None:
            print(name, key, "hit.")
        return obj

    def dump(self, dumper, name, key, obj):
        self.buffer[name][key] = obj
        path = Path(self.folder, name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        dumper(obj, path)
        if self.verbose:
            print(name, key, "cached.")

    def __call__(self, key=urlencode, dumper=pickle_dumper, loader=pickle_loader):
        if dumper is None:
            dumper = dumbrw

        if loader is None:
            loader = dumbrw

        def wrapper(func):
            def wrapped(*args, **kwargs):
                argnames = inspect.getargspec(func)[0][: len(args)]
                kwargs = {**kwargs, **dict(zip(argnames, args))}
                s = key(kwargs)
                obj = self.load(loader, func.__name__, s)
                if obj is None:
                    obj = func(**kwargs)
                    self.dump(dumper, func.__name__, s, obj)
                return obj

            return wrapped

        return wrapper
