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
os.environ['TORCH_HOME'] = "/media/hdd/Datasets/"
import sys
sys.path.append("/media/hdd/github/sprintdl/")
# -

from sprintdl.main import *
import sprintdl

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

# # Define required

# +
fpath = Path("/media/hdd/Datasets/DenseHaze/")

# train_transform = [A.Resize(128,128)]

# tfms = [ATransform(train_transform, c_in = 3)]
image_size = 64
tfms = [make_rgb,to_byte_tensor,to_float_tensor, ResizeFixed(image_size)]
bs = 64
# -

fpath.ls()

fpath_ims = fpath/"train"
fpath_mask = fpath/"masks"


# # Actual process

def return_mask(name):
    mask = from_another_folder(fpath_ims, fpath_mask,name, replace_fun=lambda x:x)
    return mask


class DoubleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_func= lambda x: x, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.items = Path.ls(image_dir)
        self.path = image_dir
        self.label_func = label_func

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        img_path = self.image_dir/self.items[index]
        mask_path = self.label_func(img_path)
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        image = compose(image, self.transform[0])
        mask = compose(mask , self.transform[1])
        return image, mask


def double_data(fpath_ims, fpath_mask,label_func = lambda x: x, c_in=3, c_out = 3, bs = 64, num_workers=8, pct_split =.2, transform_im= None, transform_ma= None, pin_memory=True):
    il = DoubleDataset(fpath_ims, fpath_mask, transform= [transform_im, transform_ma])
#     return il
    tot = len(il)
    split = int(pct_split*tot)
    train_ds, val_ds = torch.utils.data.random_split(il, (split, tot-split))
    
    train_loader = torch.utils.data.DataLoader(
        train_ds.dataset,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds.dataset,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return DataBunch(train_loader, val_loader, c_in=c_in, c_out=c_out)


data = double_data(fpath_ims, fpath_mask,c_in = 3, c_out = 3, transform_im= tfms, transform_ma= tfms, label_func = return_mask)

# # Training

# +
lr = 1e-4
pct_start = 0.3
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

# def loss_funct(input, target):
# #     print(input.shape, target.shape)
#     return nn.BCEWithLogitsLoss()(input, target) 
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def loss_funct(pred, target, bce_weight=0.5):
    pred = torch.cat((pred, pred, pred), dim = 1)
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

cbfs = [
    partial(AvgStatsCallback,loss_funct),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
#     MixUp,
       partial(CudaCallback, device)]

loss_func=loss_funct
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)


# +
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torchvision.transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# -

arch = UNET(3)
clear_memory()
learn = Learner(arch, data, loss_funct, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)

learn.fit(10)

open_image("/media/hdd/Datasets/boat/buoy/alaska-ocean-warning-light-sea-2574393.jpg",(128,128))

predict_image(learn, (500,500), "/media/hdd/Datasets/DenseHaze/train/01.png", convert_to="RGB")

open_image("/home/eragon/Downloads/2020-Ferrari-F8-Tributo.jpg",(256,128))

predict_image(learn, (256,128), "/home/eragon/Downloads/2020-Ferrari-F8-Tributo.jpg", convert_to="RGB",plot = True)

learn.recorder.plot_loss()

learn.recorder.plot_lr()














