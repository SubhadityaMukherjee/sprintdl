import gzip
import pickle
from collections import OrderedDict
from typing import *

import torch
from torch import nn, tensor


def normalize(x, m, s):
    return (x - m) / s


def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def normalize_to(train, valid):
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)


def flatten(x):
    return x.view(x.shape[0], -1)


def mnist_resize(x):
    return x.view(-1, 1, 28, 28)


def get_hist(h):
    return torch.stack(h.stats[2]).t().float().log1p()


def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[19:22].sum(0) / h1.sum(0)


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx) == len(self)  # bool mask
            return [o for m, o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


def get_batch(dl):
    dl.xb, dl.yb = next(iter(dl))
    for cb in dl.cbs:
        cb.set_runner(dl)
    dl("begin_batch")
    return dl.xb, dl.yb


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(l, lin_layers)


def append_stat(hook, mod, inp, outp):
    d = outp.data
    hook.mean, hook.std = d.mean().item(), d.std().item()


def get_mnist(path):
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        tensor, (x_train, y_train, x_valid, y_valid)
    )
    return x_train, y_train, x_valid, y_valid


def untar_data(path):  # TODO : get and untar data from URL or path. Return path
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _)
    x_train, y_train, x_valid, y_valid = map(
        tensor, (x_train, y_train, x_valid, y_valid)
    )
    return x_train, y_train, x_valid, y_valid


def setify(i):
    return i if isinstance(i, set) else set(listify(i))


def compose(x, funcs, *args, order_key="_order", **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res


def show_image(im, figsize=(3, 3)):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(im.permute(1, 2, 0))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def lin_comb(v1, v2, beta):
    return beta * v1 + (1 - beta) * v2


def param_getter(m):
    return m.parameters()
