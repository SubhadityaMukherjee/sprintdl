import operator

import torch


def test(a, b, cmp, cname=None):
    if cname is None:
        cname = cmp.__name__
        assert cmp(a, b), f"{cname}: \n{a}\n{b}"


def test_eq(a, b):
    test(a, b, operator.eq, "==")


def near(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b):
    test(a, b, near)
