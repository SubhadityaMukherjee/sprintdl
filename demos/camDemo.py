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

from sprintdl.models.xresnet import *

# # Define required

# +
fpath = Path("/media/hdd/Datasets/imagewoof2-160/")

# train_transform = [A.Resize(128,128)]

# tfms = [ATransform(train_transform, c_in = 3)]
tfms = [make_rgb,to_byte_tensor,to_float_tensor, ResizeFixed(128)]
bs = 256
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

il

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-2], proc_y=CategoryProcessor())

n_classes = len(set(ll.train.y.items));n_classes

data = ll.to_databunch(bs, c_in=3, c_out=n_classes)

show_batch(data, 4)

# # Training

# +
lr = .001
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
#     MixUp,
       partial(CudaCallback, device)]

loss_func=LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)


# -

class Net(nn.Module):
    def __init__(self,c):
        super(Net,self).__init__()
        #img = images
        self.c = c
        self.fc=nn.Linear(512,self.c)

    
    def forward(self,x):
        print(x.shape)
        to_v = x.shape[-1]
        x=x.view(512,to_v*to_v).mean(1).view(1,-1)
        x=self.fc(x)
        return  F.softmax(x,dim=1)

arch = partial(xresnet18, c_out =n_classes)()

mod = nn.Sequential(*list(arch.children())[:-3])

new_mod= nn.Sequential(mod, Net(3)).to(device)

clear_memory()
learn = Learner(new_mod, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)

params = list(learn.model.parameters())
weight = np.squeeze(params[-1].data.cpu().numpy())

weight.shape

test_im = "/media/hdd/Datasets/imagewoof2-160/val/n02086240/ILSVRC2012_val_00002701.JPEG"

test_im = open_image(fpath = test_im, size = (128,128))

test_im

test_im = compose(test_im, tfms)

logit = learn.model(test_im.unsqueeze(0).to(device))

hx = F.softmax(logit, dim = 1).data.squeeze()

hx

probs, idx = hx.sort(0, True)
probs = probs.detach().cpu().numpy()

probs, idx

features_blobs = new_mod(test_im.unsqueeze(0).to(device))
features_blobs1 = features_blobs.squeeze().cpu().detach().numpy()

features_blobs1.shape


def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
#     print(feature_conv.shape)
    bz, nc, h, w = feature_conv.shape
#     bz = 1
#     nc = 1
#     h, w = 512, 1
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        print(beforeDot.shape, weight.shape)
        cam = np.matmul(weight, beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


return_CAM(features_blobs1, weight, [idx[0]])














