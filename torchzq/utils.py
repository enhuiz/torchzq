from pathlib import Path
from natsort import natsorted
from functools import partial, cached_property
from types import GenericAlias


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


_NOT_FOUND = object()


class _delegated_cached_property(cached_property):
    def __init__(self, delegatee, func):
        self.delegatee = delegatee
        super().__init__(func)

    def get_delegatee(self, instance):
        return getattr(self, instance, None)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = self.get_delegatee(instance).__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val

    def __delete__(self, instance):
        cache = self.get_delegatee(instance).__dict__
        if self.attrname in cache:
            del cache[self.attrname]
        else:
            raise AttributeError(self.attrname)

    __class_getitem__ = classmethod(GenericAlias)


def delegated_cached_property(delegatee):
    return partial(_delegated_cached_property, delegatee)
