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
# -

from sprintdl.main import *
from sprintdl.nets import *

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

# # Define required

# +
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"

fpath = download_and_check(url, fpath = "/media/hdd/Datasets/", name = "imagewoof")
# -

fpath = untar_data(fpath)

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256

# # Define model and data

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
#     MixUp,
       partial(CudaCallback, device)]

loss_func=LabelSmoothingCrossEntropy()
arch = partial(xresnet18, c_out=10)
epochs = 5
lr = .4
# opt_func = partial(sgd_mom_opt, wd=0.01)
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# opt_func = lamb
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

il

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

ll

data = ll.to_databunch(bs, c_in=3, c_out=10)

# # Training

# learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)
learn = Learner(arch(), data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

# +
# model_summary(learn, data)
# -

learn.fit(epochs)

# # Save model

sm1 = save_model(learn, "5epoch", fpath)

# # New data 

tfms = [make_rgb, ResizeFixed(256), to_byte_tensor, to_float_tensor]
cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
#     Recorder,
#     MixUp,
       partial(CudaCallback, device)]
bs = 64

il = ImageList.from_files(fpath, tfms=tfms)

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

data = ll.to_databunch(bs, c_in=3, c_out=10)

# learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)
learn = Learner(arch(), data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

adapt_model(learn, data, "/media/hdd/Datasets/imagewoof2-160/models/5epoch")

learn.fit(epochs)




