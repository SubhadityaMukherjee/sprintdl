# Contents
[TOC]

# sprintDL, a wrapper for Pytorch
- By Subhaditya Mukherjee
- This is a pretty huge library and there are many useful things you can do with it. So here is a handy guide of sorts.

# Function documentation
- [Site](https://subhadityamukherjee.github.io/sprintdl/)

## Imports
```python
from sprintdl.main import *
from sprintdl.nets import *

device = torch.device('cuda',0)
```
## Classification example
- The library follows these steps to the most extent.

```python
# Dataset path
fpath = Path("/media/hdd/Datasets/ArtClass/")

# Transforms
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 256

# Get all files
il = ImageList.from_files(fpath, tfms=tfms)

# Split data using random or something else
sd = SplitData.split_by_func(il, partial(random_splitter, p_valid = .2))

# Label using a function to get classes
ll = label_by_func(sd, lambda x: str(x).split("/")[-3], proc_y=CategoryProcessor())

# Get number of classes
n_classes = len(set(ll.train.y.items))

# Convert to a custom object
data = ll.to_databunch(bs, c_in=3, c_out=2)

# Show 4 examples from the dataset at random
show_batch(data, 4)

# Define learning rate
lr = .001

# Define the schedules for annealers
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

# Callbacks
# Save statistics, Schedule Params, Normalize, Pretty progress bar, enable hooks, use GPU
cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
    partial(CudaCallback, device)]

# Choose loss function
loss_func=LabelSmoothingCrossEntropy()

# Choose architecture and use number of classes
arch = partial(xresnet18, c_out=n_classes)

# Choose an optimizer function
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)

clear_memory()

# Create a learner class with all the above
learn = Learner(arch(), data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt_func)

# Get output sizes of each layer
model_summary(learn, data)

# Train for n epochs
learn.fit(3)

# Save model
save_model(learn, "m1", fpath)

# Get classification report
classification_report(learn, n_classes, device)

# Predict for a single image
temp = Path('/media/hdd/Datasets/ArtClass/Popular/artgerm/10004370_1657536534486515_1883801324_n.jpg')
get_class_pred(temp, learn ,ll, 128)

# Plot Learning rate and loss
learn.recorder.plot_lr()
learn.recorder.plot_loss()

# Plot Model visualization
run_with_act_vis(1, learn)
```


