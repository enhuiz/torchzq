from collections import defaultdict


class Event(list):
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)


def create_events(*names):
    return type("EventSystem", (), {name: Event() for name in names})
