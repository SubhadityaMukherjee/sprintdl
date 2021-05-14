import time
from functools import partial

import PIL
import torch
from torch import tensor


class Transform:
    _order = 0


def make_rgb(item):
    return item.convert("RGB")


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


_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())
