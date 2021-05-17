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

import sys
sys.path.append("../")

from sprintdl.main import *
from sprintdl.models.efficientnet import *
from sprintdl.models.xresnet import *

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math
# -

# # Define required

import pandas as pd

fpath = Path("/media/hdd/Datasets/blindness/trainLabels.csv")

df = pd.read_csv(fpath); df.head(3)

len(df.level.unique())

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256

# +
lr = 1e-2
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
       partial(CudaCallback, device)]

loss_func=LabelSmoothingCrossEntropy()
lr = .001
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# -

tl = TableLoader(fpath, "image", "level", add_before="/media/hdd/Datasets/blindness/trainImages/", add_after=".jpg")

il = ImageList(tl, tfms = tfms)

il

sd = SplitData.split_by_func(il, partial(random_splitter,p_valid = .2))

sd

ll = label_by_func(sd, partial(table_labeler, dic = tl), proc_y=CategoryProcessor())

ll

data = ll.to_databunch(bs, c_in=3, c_out=5)

show_batch(data)

arch = partial(xresnet34, c_out =5)()

learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)




