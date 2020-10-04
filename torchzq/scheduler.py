import math


class _ScheduleFunction:
    def __init__(self, epochwise):
        self.epochwise = epochwise

    def step(self, epoch, iteration):
        if self.epochwise:
            self.n = epoch
        else:
            self.n = iteration


class Cosine(_ScheduleFunction):
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


class Exponential(_ScheduleFunction):
    def __init__(self, k, epochwise):
        super().__init__(epochwise)
        self.k = k
        assert k > 0

    def __call__(self):
        return math.exp(-self.k * self.n)


class Constant(_ScheduleFunction):
    def __init__(self, c):
        super().__init__(True)
        self.c = c

    def __call__(self):
        return self.c


class Lambda(_ScheduleFunction):
    def __init__(self, f, epochwise):
        super().__init__(epochwise)
        self.f = f

    def __call__(self):
        return self.f(self.n)


class Logistic(_ScheduleFunction):
    def __init__(self, k, start, upper, epochwise=True):
        super().__init__(epochwise)
        self.k = k
        self.start = start
        self.upper = upper
        assert k > 0

    def __call__(self):
        k = self.k
        start = self.start
        upper = self.upper
        return upper / (1 + math.exp(-k * (self.n - start)))


class Scheduler:
    def __init__(self):
        self._functions = []

    def schedule(self, x):
        x = eval(str(x))
        if not callable(x):
            x = Constant(x)
        self._functions.append(x)
        return x

    def step(self, epoch, iteration):
        for function in self._functions:
            function.step(epoch, iteration)
