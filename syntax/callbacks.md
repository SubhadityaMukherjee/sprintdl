[TOC]

# Usage
```python
lr = .001
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))

# List callbacks
cbfs = [
    partial(AvgStatsCallback,accuracy),
    partial(ParamScheduler, 'lr', sched_lr),
    partial(ParamScheduler, 'mom', sched_mom),
    partial(BatchTransformXCallback, norm_imagenette),
    ProgressCallback,
    Recorder,
    partial(CudaCallback, device)]

# Pass to learner
learn = Learner(arch,  data, loss_func, lr=lr, cb_funcs=cbfs)
```

# Partial
- Use partial to change any of the class values

# Callback list
- TrainEvalCallback
- ProgressCallback
- partial(BatchTransformXCallback, norm_imagenette)
- partial(BatchTransformXCallback, norm_mnist)
- CudaCallback
- LR_Find
- partial(ParamScheduler, 'lr', sched) 
- Recorder
- AvgStatsCallback

## List of schedulers
- sched_exp  #exponential
- cos_1cycle_anneal
- combine_scheds
- sched_no #no sched
- sched_cos #cosine
- sched_lin #linear


