import inspect
from types import LambdaType
from math import *


class _ScheduledVariable:
    def __init__(self):
        self.listeners = []

    def step(self, current_epoch, global_step):
        self.current_epoch = current_epoch
        self.global_step = global_step
        value = self()
        for listener in self.listeners:
            listener(value)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def __call__(self) -> int:
        raise NotImplementedError

    def set_repr(self, repr):
        self._repr = repr

    def __repr__(self):
        return self._repr

    def __deepcopy__(self, _):
        msg = "Warning: Deepcopying a scheduler is forbidden. Shallow copy is performed instead."
        print(msg)
        return self


class Constant(_ScheduledVariable):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def __call__(self):
        return self.c


class Lambda(_ScheduledVariable):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self):
        params = {p.name for p in inspect.signature(self.f).parameters.values()}

        def make_kwargs(ks, v):
            d = {}
            for k in ks:
                if k in params:
                    d.setdefault(k, v)
            return d

        kwargs = make_kwargs(["e", "epoch", "current_epoch"], self.current_epoch)
        kwargs |= make_kwargs(["i", "step", "global_step"], self.global_step)

        return self.f(**kwargs)


class Scheduler:
    def __init__(self):
        self._functions = []

    def schedule(self, s):
        x = eval(str(s))
        if isinstance(x, LambdaType):
            x = Lambda(x)
        elif not callable(x):
            x = Constant(x)
        x.set_repr(s)
        self._functions.append(x)
        return x

    def step(self, current_epoch, global_step):
        for function in self._functions:
            function.step(current_epoch, global_step)
