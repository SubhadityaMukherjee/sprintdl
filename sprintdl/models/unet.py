import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..layers import *


def _get_sfs_idxs(sizes):
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0]
    )
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class UnetBlock(nn.Module):
    def __init__(
        self,
        up_in_c,
        x_in_c,
        hook,
        final_div=True,
        blur=False,
        leaky=None,
        self_attention=False,
        **kwargs
    ):
        self.hook = hook
        self.shuf = PixelShuffleICNR(
            up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c, NormType.Batch)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(
            nf, nf, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
