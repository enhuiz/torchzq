import signal
import contextlib
from multiprocessing import parent_process


def mutex(f):
    occupied = False

    def wrapped(*args, **kwargs):
        nonlocal occupied
        if not occupied:
            occupied = True
            f(*args, **kwargs)
            occupied = False

    return wrapped


@contextlib.contextmanager
def graceful_interrupt_handler(
    signums=[signal.SIGINT, signal.SIGQUIT],
    callback=lambda _: None,
):
    callback = mutex(callback)
    for signum in signums:
        signal.signal(signum, lambda signum, _: parent_process() or callback(signum))
    yield
    for signum in signums:
        signal.signal(signum, signal.SIG_DFL)
