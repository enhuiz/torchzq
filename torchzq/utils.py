from pathlib import Path
from natsort import natsorted


class EMAMeter:
    def __init__(self, decay=0.98):
        self.decay = decay
        self.value = None

    def __call__(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value


def print_directory_tree(root: Path, prefix: str = ""):
    if not root.exists():
        return
    print(f"{prefix}{root.name if prefix else root}")
    if root.is_dir():
        base = prefix.replace("─", " ").replace("├", "│").replace("└", " ")
        paths = natsorted(root.iterdir())
        for i, path in enumerate(paths):
            if i < len(paths) - 1:
                print_directory_tree(path, base + "├── ")
            else:
                print_directory_tree(paths[-1], base + "└── ")
