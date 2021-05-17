import os

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
import sys

sys.path.append("../")
# -

from sprintdl.main import *

device = torch.device("cuda", 0)
import torch

# # Define required

# +
fpath = Path("/media/hdd/Datasets/ArtClass/")

tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256
# -

# # Actual process

il = ImageList.from_files(fpath, tfms=tfms)

tm = Path(
    "/media/hdd/Datasets/ArtClass/Unpopular/mimang.art/69030963_140928767119437_3621699865915593113_n.jpg"
)


sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.2))
ll = label_by_func(sd, lambda x: str(x).split("/")[-3], proc_y=CategoryProcessor())

n_classes = len(set(ll.train.y.items))

data = ll.to_databunch(bs, c_in=3, c_out=2)

lr = 0.001
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr = combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

cbfs = [
    partial(AvgStatsCallback, accuracy),
    partial(ParamScheduler, "lr", sched_lr),
    partial(ParamScheduler, "mom", sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
    partial(CudaCallback, device),
]

loss_func = LabelSmoothingCrossEntropy()
arch = get_vision_model("resnet34", n_classes=n_classes, pretrained=True)

opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)

# # Training

clear_memory()

learn = Learner(arch, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

learn.fit(1)
