import torch

from torchzq.named_array import NamedArray


def test():
    a = torch.randn(5, requires_grad=True)
    b = torch.randn(5)

    na = NamedArray(dict(zip("abcde", a)))
    nb = NamedArray(dict(zip("abcde", b)))

    assert na.values().requires_grad

    indices = [3, 4, 2, 1, 0]
    snb = NamedArray(dict(zip(["abcde"[i] for i in indices], b[indices])))
    print(snb)

    assert ((0 + nb).values() == (0 + b)).all()
    assert ((0 - nb).values() == (0 - b)).all()
    assert ((1 * nb).values() == (1 * b)).all()
    assert ((1 / nb).values() == (1 / b)).all()

    assert ((na + nb).values() == (a + b)).all()
    assert ((na - nb).values() == (a - b)).all()
    assert ((na * nb).values() == (a * b)).all()
    assert ((na / nb).values() == (a / b)).all()

    assert ((na + snb).values() == (a + b)).all()
    assert ((na - snb).values() == (a - b)).all()
    assert ((na * snb).values() == (a * b)).all()
    assert ((na / snb).values() == (a / b)).all()

    assert ((na + 2.33) == (a + 2.33)).all()
    assert ((na - 2.33) == (a - 2.33)).all()
    assert ((na * 2.33) == (a * 2.33)).all()
    assert ((na / 2.33) == (a / 2.33)).all()

    assert na.sum() == a.sum()
    assert nb.sum() == b.sum()
    assert nb.mean() == b.mean()
    assert nb.std() == b.std()

    na["f"] = torch.ones([1])
    a = torch.cat([a, torch.ones([1])])

    assert (na == a).all()

    na["g"] = torch.zeros([])
    a = torch.cat([a, torch.zeros([1])])
    assert (na == a).all()
