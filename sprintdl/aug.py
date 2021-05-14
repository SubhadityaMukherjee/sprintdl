import math
import random
from functools import partial

import matplotlib.pyplot as plt
import PIL
import torch
from torch import tensor

from .helpers import listify


class Transform:
    _order = 0


def make_rgb(item):
    return item.convert("RGB")


make_rgb._order = 0


class ResizeFixed(Transform):
    _order = 10

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, item):
        #  print(self.size, item.size())
        #  if torch.is_tensor(item):
        #      item = to_byte_tensor(item)
        return item.resize(self.size, PIL.Image.BILINEAR)


def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w, h = item.size
    return res.view(h, w, -1).permute(2, 0, 1)


to_byte_tensor._order = 20


def to_float_tensor(item):
    return item.float().div_(255.0)


to_float_tensor._order = 30


def normalize_chan(x, mean, std):
    #  if not torch.is_tensor(x):
    #      x = to_byte_tensor(x).cuda()
    #
    return (x - mean[..., None, None]) / std[..., None, None]


def show_image(im, ax=None, figsize=(3, 3)):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.imshow(im.permute(1, 2, 0))


def show_batch(x, c=4, r=None, figsize=None):
    n = len(x)
    if r is None:
        r = int(math.ceil(n / c))
    if figsize is None:
        figsize = (c * 3, r * 3)
    fig, axes = plt.subplots(r, c, figsize=figsize)
    for xi, ax in zip(x, axes.flat):
        show_image(xi, ax)


class PilTransform(Transform):
    _order = 11


class PilRandomFlip(PilTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random() < self.p else x


class PilRandomDihedral(PilTransform):
    def __init__(self, p=0.75):
        self.p = p * 7 / 8

        def __call__(self, x):
            if random.random() > self.p:
                return x
            return x.transpose(random.randint(0, 6))


def processs_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz) == 2 else [sz[0], sz[0]])


def default_crop_size(w, h):
    return [w, w] if w < h else [h, h]


class GeneralCrop(PilTransform):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR):
        self.resample, self.size = resample, processs_sz(size)
        self.crop_size = None if crop_size is None else processs_sz(crop_size)

    def default_crop_size(self, w, h):
        return default_crop_size(w, h)

    def __call__(self, x):
        csize = (
            self.default_crop_size(*x.size)
            if self.crop_size is None
            else self.crop_size
        )
        return x.transform(
            self.size,
            PIL.Image.EXTENT,
            self.get_corners(*x.size, *csize),
            resample=self.resample,
        )

    def get_corners(self, w, h):
        return (0, 0, w, h)


class CenterCrop(GeneralCrop):
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale

    def default_crop_size(self, w, h):
        return [w / self.scale, h / self.scale]

    def get_corners(self, w, h, wc, hc):
        return ((w - wc) // 2, (h - hc) // 2, (w - wc) // 2 + wc, (h - hc) // 2 + hc)


_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())
