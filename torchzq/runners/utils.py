import torch
import contextlib


@contextlib.contextmanager
def autocast_if(cond):
    if cond:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield
