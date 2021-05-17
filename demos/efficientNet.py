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
sys.path.append("../")
# -

from sprintdl.main import *
import sprintdl

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

try:
    from efficientnet_pytorch import EfficientNet
except:
    print("Please install efficientnet_pytorch using 'pip install efficientnet_pytorch'")


# # Define required

# +
fpath = Path("/media/hdd/Datasets/ArtClass/")

tfms = [make_rgb, ResizeFixed(64), to_byte_tensor, to_float_tensor]
bs = 256
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

il

tm= Path("/media/hdd/Datasets/ArtClass/Unpopular/mimang.art/69030963_140928767119437_3621699865915593113_n.jpg")

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-3], proc_y=CategoryProcessor())

n_classes = len(set(ll.train.y.items))

data = ll.to_databunch(bs, c_in=3, c_out=2)

show_batch(data, 4)

# # EfficientNet

eb1 = sprintdl.models.efficientnet(num_classes=n_classes, pretrained=False, name = 'efficientnet-b1')
eb2 = sprintdl.models.efficientnet(num_classes=n_classes, pretrained=False, name = 'efficientnet-b2')
eb3 = sprintdl.models.efficientnet(num_classes=n_classes, pretrained=False, name = 'efficientnet-b3')
eb4 = sprintdl.models.efficientnet(num_classes=n_classes, pretrained=False, name = 'efficientnet-b4')
eb5 = sprintdl.models.efficientnet(num_classes=n_classes, pretrained=False, name = 'efficientnet-b5')

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

dict_runner = {
    "e1":[1, eb1, data, loss_func, .001, cbfs,opt_func],
    "e2":[1, eb2, data, loss_func, .001, cbfs,opt_func],
    "e3":[1, eb3, data, loss_func, .001, cbfs,opt_func],
    "e4":[1, eb4, data, loss_func, .001, cbfs,opt_func],
    "e5":[1, eb5, data, loss_func, .001, cbfs,opt_func],
}

for i in [eb1, eb2, eb3, eb4, eb5]:
    count_parameters(i, table=False)

multiple_runner(dict_run=dict_runner, save = False)






