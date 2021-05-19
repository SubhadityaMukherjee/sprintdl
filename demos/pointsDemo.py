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
fpath = Path("/media/hdd/Datasets/faceKeypoint/")

train_transform = [
#                    A.SmallestMaxSize(max_size=160),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.RandomCrop(height=128, width=128),
        
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]

tfms = [ATransform(train_transform, c_in = 2)]
bs = 128
# -

# # Actual process

import pandas as pd

df = pd.read_csv(fpath/'training.csv')

df.head(2)


def get_locs(flname):
    index = int(flname.name[:-4])
    plist=[]
    coords=list(df.loc[index])
    for i in range(len(coords)//2):
        plist.append([coords[i*2+1],coords[i*2]])
    return tensor(plist)


get_locs(Path("/media/hdd/Datasets/faceKeypoint/trainImages/2246.jpg"))

il = ImageList.from_files(fpath, tfms=tfms)

il

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))
ll = label_by_func(sd, get_locs)

n_classes = 15

n_classes

data = ll.to_databunch(bs, c_in=3, c_out=n_classes)

data.train_ds.y.items[0]

plot_keypoints([x.permute(1,2,0) for x in data.train_ds.x[:3]], data.train_ds.y[:3])

show_batch(data, 4)


# # Training

def mseloss(input, target):
    target = target.view(target.size(0),-1)
    target[torch.isnan(target)] = 0
    return nn.MSELoss()(input, target.type(torch.FloatTensor).cuda())


# +
lr = .001
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgStatsCallback,[mseloss]),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
#     MixUp,
       partial(CudaCallback, device)]
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)


# -

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 30) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        out = self.fc1(x) 
        return out


arch= Net()

count_parameters(arch)

learn = None
clear_memory()

lr = .0001

learn = Learner(arch, data,mseloss, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)

learn.fit(20)

learn.recorder.plot_loss()

save_model(learn, "pointsTry",Path("/media/hdd/Datasets/faceKeypoint/"))

pred= predict_image(learn, "/media/hdd/Datasets/faceKeypoint/trainImages/2246.jpg")

plot_keypoints([sprintdl.aug.open_image("/media/hdd/Datasets/faceKeypoint/trainImages/2246.jpg", to_tensor=True, perm =(1,2,0) )], [pred])


