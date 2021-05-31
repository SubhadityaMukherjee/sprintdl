import concurrent
import gc
import inspect
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import SimpleNamespace

import torch
from prettytable import PrettyTable
from torch import nn, optim
from torch.functional import F

from .callbacks import *
from .helpers import *
from .optimizers import *

"""
This part has the major training loops and model definations
"""


class Module:
    """
    Base class
    """

    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception("not implemented")

    def backward(self):
        self.bwd(self.out, *self.args)


class DummySequential:
    """
    Sequential model non pytorch
    """

    def __init__(self, n_in, nh, n_out):
        self._modules = {}
        self.l1 = nn.Linear(n_in, nh)
        self.l2 = nn.Linear(nh, n_out)

    def __setattr__(self, k, v):
        if not k.startswith("_"):
            self._modules[k] = v
        super().__setattr__(k, v)

    def __repr__(self):
        return f"{self._modules}"

    def parameters(self):
        for l in self._modules.values():
            for p in l.parameters():
                yield p


class SequentialModel(nn.Module):
    """
    Main sequential
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


def param_getter(m):
    return m.parameters()


class Learner:
    """
    Main learner class which takes model, trains etc
    """

    def __init__(
        self,
        model,
        data,
        loss_func,
        opt_func=sgd_opt,
        lr=1e-2,
        splitter=param_getter,
        cbs=None,
        cb_funcs=None,
    ):
        self.model, self.data, self.loss_func, self.opt_func, self.lr, self.splitter = (
            model,
            data,
            loss_func,
            opt_func,
            lr,
            splitter,
        )
        self.c = len(set(self.data.train_ds.y.items))
        self.in_train, self.logger, self.opt = False, print, None
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs):
            self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self("begin_backward")
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        self.epochs, self.loss = epochs, tensor(0.0)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self("begin_epoch")

    def destroy(self):
        "Free the Learner internals, leaving just an empty shell that consumes no memory"

        class ZombieLearner(Learner):
            msg = "this object has been destroyed"

            def __getattr__(self, item):
                print(ZombieLearner.msg)
                return None

            def destroyed(*args, **kwargs):
                print(ZombieLearner.msg)

        attrs = [k for k in self.__dict__.keys() if not k.startswith("__")]
        for a in attrs:
            delattr(self, a)
        methods = [
            k
            for k in dir(self)
            if not k.startswith("__") and inspect.isroutine(getattr(self, k))
        ]
        for m in methods:
            setattr(self, m, ZombieLearner.destroyed)
        self.__class__ = ZombieLearner
        gc.collect()
        print(
            "this Learner object self-destroyed - it still exists, but no longer usable"
        )

    def fit(self, epochs, cbs=None, reset_opt=False):
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                if not self.do_begin_epoch(epoch):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self("begin_validate"):
                        self.all_batches()
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {
        "begin_batch",
        "after_pred",
        "after_loss",
        "begin_backward",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_epoch",
        "begin_fit",
        "begin_epoch",
        "begin_validate",
        "after_epoch",
        "after_cancel_train",
        "after_fit",
    }

    def reconstruct(self, t):
        return Image(t.float().clamp(min=0, max=1))

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


def get_dls(train_ds, valid_ds, bs, num_workers=8, **kwargs):
    """
    Return dataloaders
    """
    return (
        torch.utils.data.DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, **kwargs
        ),
        torch.utils.data.DataLoader(
            valid_ds, batch_size=bs * 2, num_workers=num_workers, **kwargs
        ),
    )


def run_with_act_vis(epochs, learn):
    """
    Run learner fit while storing model visualization such as histogram of activations
    """
    with Hooks(nn.Sequential(learn.model), append_stats) as hooks:
        learn.fit(epochs)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        for h in hooks:
            ms, ss, hs = h.stats
            ax0.plot(ms)
            ax1.plot(ss)
        plt.legend(range(6))
        ax0.set_title("means")
        ax1.set_title("std")
        plt.suptitle("Activation vis for layers")

        fig, axes = plt.subplots(2, 2, figsize=(15, 6))
        for ax, h in zip(axes.flatten(), hooks[:4]):
            ax.imshow(get_hist(h), origin="lower")
            ax.axis("off")
        plt.tight_layout()
        plt.suptitle("Histograms of activations for layers")

        fig, axes = plt.subplots(2, 2, figsize=(15, 6))
        for ax, h in zip(axes.flatten(), hooks[:4]):
            ax.plot(get_min(h))
            ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.suptitle("Min hist activations for layers")


def count_parameters(learn, table=False):
    if isinstance(learn, Learner) == True:
        learn = learn.model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in learn.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if table == True:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(
    cpus=_default_cpus, cmap="viridis", return_fig=False, silent=False
)


def parallel(func, arr: Collection, max_workers: int = None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers < 2:
        results = [
            func(o, i)
            for i, o in progress_bar(enumerate(arr), total=len(arr), leave=leave)
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func, o, i) for i, o in enumerate(arr)]
            results = []
            for f in progress_bar(
                concurrent.futures.as_completed(futures), total=len(arr), leave=leave
            ):
                results.append(f.result())
    if any([o is not None for o in results]):
        return results
