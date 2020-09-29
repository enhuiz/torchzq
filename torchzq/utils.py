class EMAMeter:
    def __init__(self, decay=0.98):
        self.decay = decay
        self.value = None

    def reset(self):
        self.value = None

    def __call__(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value


class Timer:
    def __init__(self, interval=1):
        self.interval = interval
        self.restart()

    def lap(self):
        return time.time() - self.start

    def timeup(self):
        return self.lap() > self.interval

    def restart(self):
        self.start = time.time()
