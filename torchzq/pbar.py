import tqdm
from collections import defaultdict

from torchzq.utils import Timer


def create_pbar(iterable, quiet):
    pbar = tqdm.tqdm(iterable, dynamic_ncols=True, disable=quiet)

    close = pbar.close

    create_line = lambda: tqdm.tqdm(bar_format="╰{postfix}")

    if quiet:
        items = {}
        line = create_line()

        timer = Timer(10)

        def update_line(i, s):
            items[i] = s
            if timer.timeup():
                line.set_postfix_str(", ".join(items.values()) + "\n")
                timer.restart()

        def close_all():
            close()
            line.close()

    else:
        lines = defaultdict(create_line)

        def update_line(i, s):
            lines[i].set_postfix_str(s)

        def close_all():
            close()
            for line in lines.values():
                line.close()

    pbar.update_line = update_line
    pbar.close = close_all

    return pbar
