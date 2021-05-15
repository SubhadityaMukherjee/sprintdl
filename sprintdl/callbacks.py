import math
import re
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from torch import tensor

from .helpers import *

_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def camel2snake(name):
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


class Callback:
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
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class TestCallback(Callback):
    _order = 1

    def after_step(self):
        print(self.n_iter)
        if self.n_iter == 10:
            raise CancelTrainException()


# +
class AvgStats:
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


# +
class Recorder(Callback):
    def begin_fit(self):
        self.lrs, self.losses = [0 for _ in self.opt.param_groups], []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.loss.detach().cpu())

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
    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos):
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    return start


@annealer
def sched_exp(start, end, pos):
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


def combine_scheds(pcts, scheds):
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
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
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
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class CudaCallback(Callback):
    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.to(self.device), self.yb.to(self.device)


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.xb)


def view_tfm(*size):
    def _inner(x):
        return x.view(*((-1,) + size))

    return _inner


def children(m):
    return list(m.children())


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    means, stds, hists = hook.stats
    means.append(float(outp.data.mean()))
    stds.append(float(outp.data.std()))
    hists.append(outp.data.cpu().histc(40, 0, 10))


class Hooks(ListContainer):
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


class ProgressCallback(Callback):
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
    phases = listify(phases)
    return phases + [1 - sum(phases)]


def sched_1cycle(lr, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr = combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5))
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler("lr", sched_lr), ParamScheduler("mom", sched_mom)]
