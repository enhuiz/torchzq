import pandas as pd
import torch
import collections
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


def make_grad_dataframe(module):
    rows = []
    for name, p in module.named_parameters():
        if p.grad is not None and p.grad.numel() > 0:
            row = dict(
                name=name,
                min=p.grad.min().item(),
                max=p.grad.max().item(),
                mean=p.grad.mean().item(),
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        k = str(k)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
