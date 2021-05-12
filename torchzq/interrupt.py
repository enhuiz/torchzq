import signal
import contextlib
from multiprocessing import parent_process


@contextlib.contextmanager
def graceful_interrupt_handler(
    signums=[signal.SIGINT, signal.SIGQUIT],
    callback=lambda _: None,
):
    for signum in signums:
        signal.signal(signum, lambda signum, _: parent_process() or callback(signum))
    yield
    for signum in signums:
        signal.signal(signum, signal.SIG_DFL)
