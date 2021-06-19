# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

import os

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
import sys

sys.path.append("/media/hdd/github/sprintdl/")
# -

import sprintdl
from sprintdl.main import *

device = torch.device("cuda", 0)
import math

import torch
from torch.nn import init

from sprintdl.models.xresnet import *

# # Define required

# +
fpath = Path("/media/hdd/Datasets/DenseHaze/")

# train_transform = [A.Resize(128,128)]

# tfms = [ATransform(train_transform, c_in = 3)]
image_size = 128
tfms = [make_rgb, to_byte_tensor, to_float_tensor, ResizeFixed(image_size)]
bs = 64
# -

fpath_mask = fpath / "train"
fpath_ims = fpath / "masks"

# # Actual process

il = ImageList.from_files(fpath_ims, tfms=tfms)

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.2))


def return_mask(fpath_ims, fpath_mask, name):
    return open_image(
        from_another_folder(fpath_ims, fpath_mask, name),
        size=(image_size, image_size),
        convert_to="RGB",
        to_tensor=True,
    )


ll = label_by_func(sd, lambda x: return_mask(fpath_ims, fpath_mask, x))

# +
n_classes = len(set(ll.train.y.items))
n_classes

data = ll.to_databunch(bs, c_in=3, c_out=n_classes)
# -

show_batch(data, 4)

# # Training

# +
lr = 0.001
pct_start = 0.3
phases = create_phases(pct_start)
sched_lr = combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

# def loss_funct(input, target):
# #     print(input.shape, target.shape)
#     return nn.BCEWithLogitsLoss()(input, target)
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def loss_funct(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss


cbfs = [
    partial(AvgStatsCallback, loss_funct),
    partial(ParamScheduler, "lr", sched_lr),
    partial(ParamScheduler, "mom", sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
    #     MixUp,
    partial(CudaCallback, device),
]


loss_func = loss_funct
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)


# +
def conv_layer(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])

        self.layer0_1x1 = conv_layer(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])

        self.layer1_1x1 = conv_layer(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = conv_layer(128, 128, 1, 0)

        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = conv_layer(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = conv_layer(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up3 = conv_layer(256 + 512, 512, 3, 1)
        self.conv_up2 = conv_layer(128 + 512, 256, 3, 1)
        self.conv_up1 = conv_layer(64 + 256, 256, 3, 1)
        self.conv_up0 = conv_layer(64 + 256, 128, 3, 1)

        self.conv_original_size0 = conv_layer(3, 64, 3, 1)
        self.conv_original_size1 = conv_layer(64, 64, 3, 1)
        self.conv_original_size2 = conv_layer(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        #         print(out.shape)
        #         out = out.permute(0, 3, 2, 1)

        return out


# -

arch = UNet(3)
clear_memory()
learn = Learner(arch, data, loss_funct, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

clear_memory()

learn.fit(20)

open_image(
    "/media/hdd/ART/refs/beings/dragons/707acb9c40491226f69b48010cb7646b.png",
    (128, 128),
)

predict_image(
    learn,
    (128, 128),
    "/media/hdd/ART/refs/beings/dragons/707acb9c40491226f69b48010cb7646b.png",
    convert_to="RGB",
    plot=True,
)

learn.recorder.plot_loss()

learn.recorder.plot_lr()
