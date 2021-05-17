import math
import random
from functools import partial

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from albumentations.pytorch import ToTensorV2
from torch import tensor
from torch.distributions import Beta

from .callbacks import Callback
from .helpers import lin_comb, listify, unsqueeze

"""
This module takes care of the augumentations
"""


class Transform:
    """
    Base transform class
    """

    _order = 0


def make_rgb(item):
    return item.convert("RGB")


make_rgb._order = 0


def pathToTensor(im):
    return np.array(PIL.Image.open(im))


class ATransform:
    def __init__(self, t_list, c_in=3):
        self.t_list = t_list
        if c_in == 3:
            self.t_list.append(
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )
        else:
            self.t_list.append(A.Normalize(mean=(0.1307,), std=(0.3081,)))
        self.t_list.append(ToTensorV2())

    def __call__(self, item):
        tfs = A.Compose(self.t_list)(image=np.array(item))["image"]
        return tfs


class ResizeFixed(Transform):
    """
    To a fixed size
    """

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
    """
    Convert
    """
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w, h = item.size
    return res.view(h, w, -1).permute(2, 0, 1)


to_byte_tensor._order = 20


def to_float_tensor(item):
    """
    Convert
    """
    return item.float().div_(255.0)


to_float_tensor._order = 30


def normalize_chan(x, mean, std):
    """
    For 3 channel images (general ones)
    """
    #  if not torch.is_tensor(x):
    #      x = to_byte_tensor(x).cuda()
    #
    return (x - mean[..., None, None]) / std[..., None, None]


def show_image(im, ax=None, figsize=(3, 3)):
    """
    Show single image
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.imshow(im.permute(1, 2, 0))


def show_batch(data, n=4, c=4, r=None, figsize=None):
    """
    Show a batch of n images from the train dataloader
    """
    x = data.train_ds.x[:n]
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
    """
    Transform
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random() < self.p else x


class PilRandomDihedral(PilTransform):
    """
    Transform
    """

    def __init__(self, p=0.75):
        self.p = p * 7 / 8

        def __call__(self, x):
            if random.random() > self.p:
                return x
            return x.transpose(random.randint(0, 6))


def processs_sz(sz):
    """
    Get size and format it to tuple
    """
    sz = listify(sz)
    return tuple(sz if len(sz) == 2 else [sz[0], sz[0]])


def default_crop_size(w, h):
    return [w, w] if w < h else [h, h]


class GeneralCrop(PilTransform):
    """
    Base class
    """

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


# MIXUP
class NoneReduce:
    def __init__(self, loss_func):
        self.loss_func, self.old_red = loss_func, None

    def __enter__(self):
        if hasattr(self.loss_func, "reduction"):
            self.old_red = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func
        else:
            return partial(self.loss_func, reduction="none")

    def __exit__(self, type, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, "reduction", self.old_red)


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class MixUp(Callback):
    _order = 90

    def __init__(self, α: float = 0.4):
        self.distrib = Beta(tensor([α]), tensor([α]))

    def begin_fit(self):
        self.old_loss_func, self.run.loss_func = self.run.loss_func, self.loss_func

    def begin_batch(self):
        if not self.in_train:
            return
        λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1 - λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], (1, 2, 3))
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.λ)

    def after_fit(self):
        self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss, getattr(self.old_loss_func, "reduction", "mean"))


_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())
Γ = lambda x: x.lgamma().exp()
