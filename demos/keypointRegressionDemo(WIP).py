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

# # Preprocess

import pandas as pd
import shutil
from PIL import Image
fpath = Path("/media/hdd/Datasets/faceKeypoint/")
t_path = fpath/"training.csv"
id_l = pd.read_csv(t_path)

id_l.head(3)

for c in id_l.columns:
    if(id_l[c].dtype!='object'):
        id_l[c]=id_l[c].fillna(id_l[c].median())

import torchvision
import tqdm


def save_str_img(strimg,w,h,flpath):
    px=255-np.array(strimg.split(),dtype=float)
    if(len(px)==w*h and len(px)%w==0 and len(px)%h==0):
        cpx = list(px.reshape(w,h))
        img = torchvision.transforms.functional.to_pil_image(tensor([cpx,cpx,cpx]))
        img.save(flpath)
        return img
    else:
        raise Exception("Invalid height and width")


# +
train_im_path = fpath/"trainImages"

train_im_path.mkdir(exist_ok=True)
# -

id_l.shape

for index, train_row in tqdm.tqdm(id_l.iterrows(), total = id_l.shape[0]):
    save_str_img(train_row.Image,96,96,train_im_path/(str(index)+'.jpg'))


class PreProcessor():
    "Basic class for a processor that will be applied to items at the end of the data block API."
    def __init__(self, ds:Collection=None):  self.ref_ds = ds
    def process_one(self, item:Any):         return item
    def process(self, ds:Collection):        ds.items = array([self.process_one(item) for item in ds.items])



# +
class PointsProcessor(PreProcessor):
    "`PreProcessor` that stores the number of targets for point regression."
    def __init__(self, ds:ItemList): self.c = len(ds.items[0].reshape(-1))
    def process(self, ds:ItemList):  ds.c = self.c

class PointsLabelList(ItemList):
    "`ItemList` for points."
    _processor = PointsProcessor
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func = MSELossFlat()

    def get(self, i):
        o = super().get(i)
        return ImagePoints(FlowField(_get_size(self.x,i), o), scale=True)

    def analyze_pred(self, pred, thresh:float=0.5): return pred.view(-1,2)
    def reconstruct(self, t, x): return ImagePoints(FlowField(x.size, t), scale=False)

class PointsItemList(ImageList):
    "`ItemList` for `Image` to `ImagePoints` tasks."
    _label_cls,_square_show_res = PointsLabelList,False


# -

# # Define required

def pilToTensor(item):
    return torchvision.transforms.functional.pil_to_tensor(test)


# tfms = [make_rgb, to_byte_tensor, to_float_tensor, pilToTensor]
tfms = [make_rgb, pilToTensor]
bs = 128


def mloss(y_true, y_pred):
    y_true=y_true.view(-1,15,2)
    
    y_true[:,:,0]=y_true[:,:,0].clone()-y_pred[:,:,0]
    y_true[:,:,1]=y_true[:,:,1].clone()-y_pred[:,:,1]
    
    y_true[:,:,0]=y_true[:,:,0].clone()**2
    y_true[:,:,1]=y_true[:,:,1].clone()**2
    
    return y_true.sum(dim=2).sum(dim=1).sum()



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

loss_func=mloss
lr = .001
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# -

data = PointsItemList.from_files(train_im_path, tfms = tfms)

data

# +
# il = ImageList.from_files("/media/hdd/Datasets/imagewoof2-160/", tfms = tfms)

# +
# il
# -

sd = SplitData.split_by_func(data, partial(random_splitter,p_valid = .2))

sd


def get_locs(flname):
    index = int(flname.name[:-4])
    plist=[]
    coords=list(id_l.loc[index])
    for i in range(len(coords)//2):
        plist.append([coords[i*2+1],coords[i*2]])
    return tensor(plist)


get_locs(Path("/media/hdd/Datasets/faceKeypoint/trainImages/2246.jpg"))

# +
# sd.train.items
# -

ll = label_by_func(sd, get_locs)

ll.train.x

data = ll.to_databunch(bs, c_in=3, c_out=10)


def show_image(im, ax=None, figsize=(3, 3)):
    """
    Show single image
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.imshow(im.permute(1, 2, 0))


def show_batch(data, n=4, c=4, r=None, figsize=None):
    """
    Show a batch of n images from the train dataloader
    """
    x = data.train_ds.x[:n]
    if r is None:
        r = int(math.ceil(n / c))
    if figsize is None:
        figsize = (c * 3, r * 3)
    fig, axes = plt.subplots(r, c, figsize=figsize)
    for xi, ax in zip(x, axes.flat):
#         xi = torchvision.transforms.functional.pil_to_tensor(xi)
        show_image(xi, ax)


show_batch(data)

learn = Learner(get_vision_model('resnet34',1), data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)




