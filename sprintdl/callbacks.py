import math
import re
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils.prune as prune
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from torch import tensor
from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import OptState
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .helpers import *

writer = SummaryWriter()


def normalize_chan(x, mean, std):
    """
    For 3 channel images (general ones)
    """
    x = x.cuda()
    return (x - mean[..., None, None]) / std[..., None, None]


_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


"""
This module handles all the cool features that require callbacks
"""

_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def camel2snake(name):
    """
    Camel case -> Snake case
    """
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


class Callback:
    """
    Base class
    """

    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    """
    Run a train and eval loop to see if everything is okay
    """

    def begin_fit(self):
        self.run.n_epochs = 0.0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        try:
            self.model.train()
        except:
            self.model_D.train()
            self.model_G.train()
        self.run.in_train = True

    def begin_validate(self):
        try:
            self.model.eval()
        except:
            self.model_D.train()
            self.model_G.train()
        self.run.in_train = False


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class TestCallback(Callback):
    """
    To see if one batch works
    """

    _order = 1

    def after_step(self):
        print(self.n_iter)
        if self.n_iter == 10:
            raise CancelTrainException()


# +
class AvgStats:
    """
    Store statistics of training
    """

    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.0, 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgStatsCallback(Callback):
    """
    Main callback for stats. Used for the progress bar
    """

    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(
            metrics, False
        )

    def begin_fit(self):
        met_names = ["loss"] + [m.__name__ for m in self.train_stats.metrics]
        names = (
            ["epoch"]
            + [f"train_{n}" for n in met_names]
            + [f"valid_{n}" for n in met_names]
            + ["time"]
        )
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class TensorboardCallback(Callback):
    """
    Main callback for tensorboard.
    """

    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(
            metrics, False
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def time_seconds(self, x):
        return sum([a * b for a, b in zip([3600, 60, 1], map(int, x.split(":")))])

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        writer.add_scalar("Loss/train", float(stats[1]), int(self.epoch))
        writer.add_scalar("Accuracy/train", float(stats[2]), int(self.epoch))
        writer.add_scalar("Loss/valid", float(stats[3]), int(self.epoch))
        writer.add_scalar("Accuracy/train", float(stats[4]), int(self.epoch))
        writer.add_scalar(
            "Time/epoch", float(self.time_seconds(str(stats[5]))), int(self.epoch)
        )


# +
class Recorder(Callback):
    """
    Stores some values that will enable visualization and hooks to get model state
    """

    def begin_fit(self):
        self.lrs, self.losses, self.nb_batches = (
            [0 for _ in self.opt.param_groups],
            [],
            [],
        )

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.loss.detach().cpu())

    def after_epoch(self):
        self.nb_batches.append(self.epoch)

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_loss(self):
        plt.plot(self.losses[: len(self.losses)])

    def plot(self):
        losses = [o.item() for o in self.losses]
        n = len(losses)
        plt.xscale("log")
        plt.plot(self.lrs[:n], losses[:n])


class ParamScheduler(Callback):
    """
    Schedules parameters that were requested
    """

    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs * len(self.opt.param_groups)
        pos = self.n_epochs / self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)


def annealer(f):
    """
    Base anneal class
    """

    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos):
    """
    Linear schedule
    """
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    """
    Cosine schedule
    """

    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    """
    No schedule
    """

    return start


@annealer
def sched_exp(start, end, pos):
    """
    Exponential Scheduling
    """
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    """
    One cycle scheduling
    """
    return [sched_cos(start, high), sched_cos(high, end)]


def combine_scheds(pcts, scheds):
    """
    Merge multiple schedulers
    """
    assert sum(pcts) == 1.0
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero(as_tuple=False).max()
        if idx == 2:
            idx = 1
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner


class LR_Find(Callback):
    """
    Find Learning rate.
    """

    def __init__(self, max_iter=100, min_lr=1e-7, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.hypers:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            print(f"Best loss: {self.best_loss}")
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class CudaCallback(Callback):
    """
    Push to cuda
    """

    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        try:
            self.model.to(self.device)
        except:
            self.model_D.to(self.device)
            self.model_G.to(self.device)

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.to(self.device), self.yb.to(self.device)


class FP16(Callback):  # TODO
    """
    FP16
    """

    _order = 4

    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        with torch.cuda.amp.autocast():
            self.model.to(self.device)

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.to(self.device), self.yb.to(self.device)

    def fpstep(self, opt):
        for p, hyper in opt.grad_params():
            scaler.step(p)

    def after_fit(self):
        scaler.scale(self.loss)
        self.fpstep(self.opt)
        scaler.update()


class BatchTransformXCallback(Callback):
    """
    Perform batch transforms
    """

    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.xb)


def view_tfm(*size):
    """
    Grab transforms
    """

    def _inner(x):
        return x.view(*((-1,) + size))

    return _inner


def children(m):
    """
    Model children
    """
    return list(m.children())


class Hook:
    """
    More base
    """

    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_stats(hook, mod, inp, outp):
    """
    Grab stats
    """
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    means, stds, hists = hook.stats
    means.append(float(outp.data.mean()))
    stds.append(float(outp.data.std()))
    hists.append(outp.data.cpu().histc(40, 0, 10))


class Hooks(ListContainer):
    """
    Base class
    """

    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


def measure_module_sparsity(module, weight=True, bias=False, use_mask=True):
    """https://leimao.github.io/blog/PyTorch-Pruning/"""

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(
    model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False
):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask
            )
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask
            )
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


conv_names = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)


def pruner(
    model, grouped_pruning=True, conv2d_prune_amount=0.4, linear_prune_amount=0.2
):
    if grouped_pruning == True:
        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if any([isinstance(module, x) for x in conv_names]):
                parameters_to_prune.append((module, "weight"))
        print(f"Pruning: {len(parameters_to_prune)}")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=conv2d_prune_amount,
        )
    else:
        for module_name, module in model.named_modules():
            if any([isinstance(module, x) for x in conv_names]):
                prune.l1_unstructured(module, name="weight", amount=conv2d_prune_amount)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)


class PruningCallback(Callback):
    def __init__(
        self, grouped_pruning=True, conv2d_prune_amount=0.4, linear_prune_amount=0.2
    ):
        self.grouped_pruning = grouped_pruning
        self.conv2d_prune_amount = conv2d_prune_amount
        self.linear_prune_amount = linear_prune_amount

    def begin_epoch(self):
        pruner(
            self.model,
            self.grouped_pruning,
            self.conv2d_prune_amount,
            self.linear_prune_amount,
        )


class ProgressCallback(Callback):
    """
    Pretty progress bar using fastprogress
    """

    _order = -1

    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)

    def begin_epoch(self):
        self.set_pb()

    def begin_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)


def create_phases(phases):
    """
    For one cycle
    """
    phases = listify(phases)
    return phases + [1 - sum(phases)]


def sched_1cycle(lr, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    """
    One cycle scheduling
    """
    phases = create_phases(pct_start)
    sched_lr = combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5))
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler("lr", sched_lr), ParamScheduler("mom", sched_mom)]


def lr_finder(learn, n_epochs, set_suggested=True):
    """
    Get suggested_lr and return
    """
    learn.cbs.append(LR_Find)
    learn.fit(n_epochs)
    suggested_lr = learn.opt.hypers[0]["lr"]
    print(f"Suggested lr : {suggested_lr}")
    learn.cbs.remove(LR_Find)
    if set_suggested == True:
        learn.lr = suggested_lr
    print(f"Set lr to suggested_lr")
    return suggested_lr


class GradientClipping(Callback):
    "Gradient clipping during training."

    def __init__(self, clip: float = 0.0):
        super().__init__()
        self.clip = clip

    def on_backward_end(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip:
            nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)


def clip_grad(learn, clip: float = 0.1):
    "Add gradient clipping of `clip` during training."
    learn.callback_fns.append(partial(GradientClipping, clip=clip))
    return learn


class MixUp(Callback):
    "Callback that creates the mixed-up input and target."

    def __init__(self, alpha: float = 0.4, stack_x: bool = False, stack_y: bool = True):
        super().__init__()
        self.alpha, self.stack_x, self.stack_y = alpha, stack_x, stack_y

    def on_train_begin(self, **kwargs):
        if self.stack_y:
            self.learn.loss_func = MixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train:
            return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else:
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = last_input * lambd.view(out_shape) + x1 * (1 - lambd).view(
                out_shape
            )
        if self.stack_y:
            new_target = torch.cat(
                [
                    last_target[:, None].float(),
                    y1[:, None].float(),
                    lambd[:, None].float(),
                ],
                1,
            )
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1 - lambd)
        return {"last_input": new_input, "last_target": new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y:
            self.learn.loss_func = self.learn.loss_func.get_old()


class MixUpLoss(nn.Module):
    "Adapt the loss function `crit` to go with mixup."

    def __init__(self, crit, reduction="mean"):
        super().__init__()
        if hasattr(crit, "reduction"):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, "reduction", "none")
        else:
            self.crit = partial(crit, reduction="none")
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(
                output, target[:, 1].long()
            )
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == "mean":
            return d.mean()
        elif self.reduction == "sum":
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, "old_crit"):
            return self.old_crit
        elif hasattr(self, "old_red"):
            setattr(self.crit, "reduction", self.old_red)
            return self.crit


class OverSampling(Callback):
    def __init__(self, weights: torch.Tensor = None):
        super().__init__()
        self.weights = weights

    def on_train_begin(self, **kwargs):
        self.old_dl = self.data.train_dl
        self.labels = self.data.train_dl.y.items
        assert np.issubdtype(
            self.labels.dtype, np.integer
        ), "Can only oversample integer values"
        _, self.label_counts = np.unique(self.labels, return_counts=True)
        if self.weights is None:
            self.weights = torch.DoubleTensor((1 / self.label_counts)[self.labels])
        self.total_len_oversample = int(self.data.c * np.max(self.label_counts))
        sampler = WeightedRandomSampler(self.weights, self.total_len_oversample)
        self.data.train_dl = self.data.train_dl.new(shuffle=False, sampler=sampler)

    def on_train_end(self, **kwargs):
        "Reset dataloader to its original state"
        self.data.train_dl = self.old_dl


class UnderSampling(Callback):
    def __init__(self, weights: torch.Tensor = None):
        super().__init__()
        self.weights = weights

    def on_train_begin(self, **kwargs):
        self.old_dl = self.data.train_dl
        self.labels = self.data.train_dl.y.items
        assert np.issubdtype(
            self.labels.dtype, np.integer
        ), "Can only undersample integer values"
        _, self.label_counts = np.unique(self.labels, return_counts=True)
        if self.weights is None:
            self.weights = torch.DoubleTensor((1 / self.label_counts)[self.labels])
        self.total_len_undersample = int(self.data.c * np.min(self.label_counts))
        sampler = WeightedRandomSampler(self.weights, self.total_len_undersample)
        self.data.train_dl = self.data.train_dl.new(shuffle=False, sampler=sampler)

    def on_train_end(self, **kwargs):
        "Reset dataloader to its original state"
        self.data.train_dl = self.old_dl


class StopAfterNBatches(Callback):
    def __init__(self, n_batches=2):
        self.stop, self.n_batches = False, n_batches - 1

    def on_batch_end(self, iteration, **kwargs):
        if iteration == self.n_batches:
            return {"stop_epoch": True, "stop_training": True, "skip_validate": True}


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

#  def set_bn_eval(m:nn.Module, use_eval=False)->None:
#      "Set bn layers in eval mode for all recursive children of `m`."
#      for l in m.children():
#          if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
#              if use_eval: l.eval()
#              else:        l.requires_grad = False
#          set_bn_eval(l)
#
#  def set_eval_except_bn(m:nn.Module, use_eval=False)->None:
#      "Set bn layers in eval mode for all recursive children of `m`."
#      for l in m.children():
#          if not isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
#              if use_eval: l.eval()
#              else:        l.requires_grad = False
#          set_eval_except_bn(l)
#
#
#  class BnFreeze(Callback):
#      run_after=TrainEvalCallback
#      "Freeze moving average statistics in all non-trainable batchnorm layers."
#      def before_train(self):
#          set_bn_eval(self.model)
#
#  class FreezeNotBn(Callback):
#      run_after=TrainEvalCallback
#      "Freeze moving average statistics in all non-trainable batchnorm layers."
#      def before_train(self):
#          set_eval_except_bn(self.model)
#
