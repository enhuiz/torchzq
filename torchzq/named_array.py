import torch
from collections.abc import Mapping


class _Operator:
    def __set_name__(self, owner, name):
        self.operator = name

    def __get__(self, obj, objtype=None):
        def wrapped(other):
            values = getattr(obj._values, self.operator)(obj._get_values(other))
            return NamedArray(dict(zip(obj.keys(), values)))

        return wrapped


class _Method:
    def __set_name__(self, owner, name):
        self.method = name

    def __get__(self, obj, objtype=None):
        return getattr(obj._values, self.method)


class NamedArray(Mapping):
    __add__ = _Operator()
    __div__ = _Operator()
    __mul__ = _Operator()
    __sub__ = _Operator()
    __add__ = _Operator()
    __div__ = _Operator()
    __mul__ = _Operator()
    __sub__ = _Operator()
    __truediv__ = _Operator()
    __floordiv__ = _Operator()
    __radd__ = _Operator()
    __rdiv__ = _Operator()
    __rmul__ = _Operator()
    __rsub__ = _Operator()
    __radd__ = _Operator()
    __rdiv__ = _Operator()
    __rmul__ = _Operator()
    __rsub__ = _Operator()
    __rtruediv__ = _Operator()
    __rfloordiv__ = _Operator()
    __eq__ = _Operator()

    sum = _Method()
    mean = _Method()
    std = _Method()
    all = _Method()
    any = _Method()

    def __init__(self, mapping=None, **kwargs):
        mapping = mapping or kwargs
        self._keys = list(mapping.keys())
        self._values = torch.stack(list(mapping.values()))

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def items(self):
        return zip(self.keys(), self.values())

    def update(self, other):
        for key in other:
            self[key] = other[key]

    def _get_values(self, other):
        if isinstance(other, NamedArray):
            return torch.stack([other[k] for k in self.keys()])
        return other

    def __setitem__(self, key, value):
        if key not in self.keys():
            self._keys.append(key)
            if value.dim() == 0:
                value = value.unsqueeze(0)
            self._values = torch.cat([self.values(), value])
        else:
            self._values[self.keys().index(key)] = value

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return self.values()[self.keys().index(key)]

    def __str__(self):
        return str(dict(sorted(zip(self.keys(), self.values()))))

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        yield from self.keys()
