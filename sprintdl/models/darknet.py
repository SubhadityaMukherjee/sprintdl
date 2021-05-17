from torch import nn

from ..layers import *


def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks // 2),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class ResLayer(nn.Module):
    def __init__(self, ni: int):
        super(ResLayer, self).__init__()
        self.conv1 = conv_bn_lrelu(ni, ni // 2, ks=1)
        self.conv2 = conv_bn_lrelu(ni // 2, ni, ks=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Darknet(nn.Module):
    def __init__(self, num_blocks: Collection[int], num_classes: int, nf=32):
        super(Darknet, self).__init__()
        layers = [conv_bn_lrelu(3, nf, ks=3, stride=1)]
        for i, nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2)
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)

    def make_group_layer(self, ch_in: int, num_blocks: int, stride: int = 1):
        return [conv_bn_lrelu(ch_in, ch_in * 2, stride=stride)] + [
            (ResLayer(ch_in * 2)) for i in range(num_blocks)
        ]

    def forward(self, x):
        return self.layers(x)
