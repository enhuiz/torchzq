import torch
from pathlib import Path
from natsort import natsorted


class ItemProperty:
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name).item()

    def __set__(self, obj, value):
        value = torch.full_like(getattr(obj, self.private_name), value)
        setattr(obj, self.private_name, value)


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


def default_tuple(x, default):
    if not isinstance(x, tuple):
        x = [x]
    else:
        x = list(x)
    for i in range(len(default) - len(x), len(default)):
        x.append(default[i])
    return tuple(x)
