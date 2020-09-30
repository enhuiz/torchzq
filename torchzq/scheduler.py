import math


class _Scheduler:
    def __init__(self, epochwise):
        self.epochwise = epochwise

    def step(self, epoch, iteration):
        if self.epochwise:
            self.n = epoch
        else:
            self.n = iteration


class Cosine(_Scheduler):
    def __init__(self, start, stop, epochwise):
        """https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay
        Args:
            start: start
            stop: end
        """
        super().__init__(epochwise)
        self.start = start
        self.stop = stop
        assert 0 <= start < stop

    def __call__(self):
        pi = math.pi
        cos = math.cos
        start = self.start
        stop = self.stop

        if self.n < start:
            return 1
        if self.n > stop:
            return 0

        return max(0, 0.5 * (1 + cos(pi * (self.n - start) / (stop - start))))


class Exponential(_Scheduler):
    def __init__(self, k, epochwise):
        super().__init__(epochwise)
        self.k = k
        assert k > 0

    def __call__(self):
        return math.exp(-self.k * self.n)


class Constant(_Scheduler):
    def __init__(self, c):
        super().__init__(True)
        self.c = c

    def __call__(self):
        return self.c


class Lambda(_Scheduler):
    def __init__(self, s, epochwise):
        super().__init__(epochwise)
        self.f = eval(s)

    def __call__(self):
        return self.f(self.n)


class SchedulerDict(dict):
    def step(self, epoch, iteration):
        for scheduler in self.values():
            scheduler.step(epoch, iteration)


def create_scheduler(x):
    x = eval(x)

    if not callable(x):
        x = Constant(x)

    return x
