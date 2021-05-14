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
# -

from sprintdl.main import *

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

# # Define required

# +
# x_train,y_train,x_valid,y_valid = get_mnist("/media/hdd/Datasets/imagenette2-160.tgz")

# +
fpath = Path("/media/hdd/Datasets/imagenette2-160/")

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256
# -

il = ImageList.from_files(fpath, tfms=tfms)

il

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

ll

data = ll.to_databunch(bs, c_in=3, c_out=10)


# # Training

# # Define model and data

# +
def prev_pow_2(x): return 2**math.floor(math.log2(x))

def get_cnn_layers(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)
    l1 = data.c_in
    l2 = prev_pow_2(l1*3*3)
    layers =  [f(l1  , l2  , stride=1),
               f(l2  , l2*2, stride=2),
               f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten), 
               nn.Linear(nfs[-1], data.c_out)]
    return layers

def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))

def get_learner(nfs, data, lr, layer, loss_func=F.cross_entropy,
                cb_funcs=None, opt_func=sgd_opt, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)


# +
nfs = [64,64,128,256]
# sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)]) 
sched = combine_scheds([0.3, 0.7], [sched_cos(.1,.3), sched_cos(.3, 0.05)])
# mnist_view = view_tfm(1,28,28)

cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched),
        partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
       partial(CudaCallback, device)]

epochs = 5
lr = .001
# opt_func = partial(sgd_mom_opt, wd=0.01)
# opt_func = adam_opt()
opt_func = lamb
# -

learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)

# +
# model_summary(learn, data)
# -

learn.fit(epochs)

run_with_act_vis(epochs,learn)

# # Digging in

learn.avg_stats.valid_stats.avg_stats

learn.recorder.plot_lr()

learn.recorder.plot_loss()
















