import inspect
from math import *


class _ScheduleFunction:
    def __init__(self):
        self.listeners = []

    def step(self, epoch, iteration):
        value = self(epoch=epoch, iteration=iteration)
        for listener in self.listeners:
            listener(value)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def __call__(self, epoch, iteration) -> int:
        raise NotImplementedError

    def set_repr(self, repr):
        self._repr = repr

    def __repr__(self):
        return self._repr

    def __deepcopy__(self, _):
        msg = "Warning: Deepcopying a scheduler is forbidden. Shallow copy is performed instead."
        print(msg)
        return self


class Constant(_ScheduleFunction):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def __call__(self, epoch, iteration):
        return self.c


class Lambda(_ScheduleFunction):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self, epoch, iteration):
        params = {p.name for p in inspect.signature(self.f).parameters.values()}

        def make_kwargs(ks, v):
            d = {}
            for k in ks:
                if k in params:
                    d.setdefault(k, v)
            return d

        return self.f(
            **(
                make_kwargs(["e", "epoch"], epoch)
                | make_kwargs(["i", "iteration"], iteration)
            )
        )


class Scheduler:
    def __init__(self):
        self._functions = []

    def schedule(self, s):
        x = eval(str(s))
        if not callable(x):
            x = Constant(x)
        x.set_repr(s)
        self._functions.append(x)
        return x

    def step(self, epoch, iteration):
        for function in self._functions:
            function.step(epoch, iteration)
