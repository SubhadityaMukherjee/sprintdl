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
from sprintdl.gans import *

device = torch.device('cuda',0)
from torch.nn import init
import torch
import math

# # Define required

# +
bs = 64
imsize = 64

fpath = Path("/media/hdd/Datasets/celeba/img_align_celeba/")

tfms = [make_rgb, ResizeFixed(imsize), to_byte_tensor, to_float_tensor]


splitter = partial(random_splitter, p_valid = .2)
label_func = lambda x: str(x).split("/")[-2]
# -

data = quick_data(fpath, tfms, splitter, label_func, bs = bs, c_in = 3,max = 1000)

show_batch(data, 4)


# # Actual

# +
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



# -

# # callbacks

# +
def genloss(x,y=None): return -torch.mean(x)

def discloss(x,y): return -(torch.mean(x) - torch.mean(y))
    


# +
lr = 5e-4
# pct_start = 0.5
# phases = create_phases(pct_start)
# sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
# sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgGANStatsCallback,genloss),
#     partial(ParamScheduler, 'lr', sched_lr),
#     partial(ParamScheduler, 'mom', sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
#     MixUp,
       partial(CudaCallback, device)]
# -

# # Training

clear_memory()

# +
latent = 1000
gen = Generator(latent, 3, imsize).cuda()
disc = Discriminator(3, imsize).cuda()
initialize_weights(gen)
initialize_weights(disc)
gen.train();
disc.train();

optd= torch.optim.RMSprop(disc.parameters(), lr=lr)
optg = torch.optim.RMSprop(gen.parameters(), lr=lr)

# +
# learn = get_learner(nfs, data, lr, conv_layer, cb_funcs=cbfs)
# learn = Learner(arch,  data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn = GANLearner(
        disc,
        gen,
        data,
        discloss,
        genloss,
        opt_func_D=optd,
        opt_func_G=optg,
        lr=lr,
        splitter=param_getter,
#         cbs=None,
        cb_funcs=cbfs,
        device="cuda:0",
        latent_dim=latent,
        clip_value=0.01,
        n_critic=5)
# -

visualize_model(learn.model_G, [1,1000,64,64])

visualize_model(learn.model_D, [1,3,64,64])

learn.fit(1)

learn.plot_output()

learn.fit(20)

learn.plot_output()

learn.fit(20)

learn.plot_output()

learn.fit(20)

learn.plot_output()

learn.fit(20)

learn.plot_output()



learn.recorder_gan.plot_lr()

learn.recorder_gan.plot_loss()

learn.plot_output()

learn.fit(30)
learn.plot_output()














