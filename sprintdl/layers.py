from pathlib import Path

import torch
from torch import nn
from torch.nn import init

from .core import *


class Flatten(nn.Module):
    """
    Flatten
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Lambda(nn.Module):
    """
    Apply a function in a model
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Lin(Module):
    """
    Linear layer
    """

    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, inp):
        return inp @ self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = torch.einsum("bi,bj->ij", inp, out.g)
        self.b.g = out.g.sum(0)


class ReLU(Module):
    def forward(self, inp):
        return inp.clamp_min(0.0) - 0.5

    def bwd(self, out, inp):
        inp.g = (inp > 0).float() * out.g


def matmul(a, b):
    """
    Matrix multiply using einsum
    """
    return torch.einsum("ik, kj-> ij", a, b)


def conv2d(ni, nf, ks=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride), nn.ReLU()
    )


class GeneralRelu(nn.Module):
    """
    General with leak or clipping if required
    """

    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def conv_rbn(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    """
    Conv relu running batchnorm
    """
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(RunningBatchNorm(nf))
    return nn.Sequential(*layers)


def init_cnn(m):
    """
    Initialize a cnn with kaiming_normal_ or constant
    """
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("vars", torch.ones(1, nf, 1, 1))
        self.register_buffer("means", torch.zeros(1, nf, 1, 1))

    def update_stats(self, x):
        m = x.mean((0, 2, 3), keepdim=True)
        v = x.mean((0, 2, 3), keepdim=True)
        self.means.lerp_(m, self.mom)
        self.vars.lerp_(v, self.mom)
        return m, v

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                m, v = self.update_stats(x)
        else:
            m, v = self.means, self.vars
        x = (x - m) / (v + self.eps).sqrt()
        return x * self.mults + self.adds


class LayerNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.0))
        self.add = nn.Parameter(tensor(0.0))

    def forward(self, x):
        m = x.mean((1, 2, 3), keepdim=True)
        v = x.var((1, 2, 3), keepdim=True)
        x = (x - m) / ((v + self.eps).sqrt())
        return x * self.mult + self.add


class InstanceNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, nf, eps=1e-0):
        super().__init__()
        self.eps = eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))

    def forward(self, x):
        m = x.mean((2, 3), keepdim=True)
        v = x.var((2, 3), keepdim=True)
        res = (x - m) / ((v + self.eps).sqrt())
        return res * self.mults + self.adds


class RunningBatchNorm(nn.Module):
    """
    Custom running batchnorm layer
    """

    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("sums", torch.zeros(1, nf, 1, 1))
        self.register_buffer("sqrs", torch.zeros(1, nf, 1, 1))
        self.register_buffer("batch", tensor(0.0))
        self.register_buffer("count", tensor(0.0))
        self.register_buffer("step", tensor(0.0))
        self.register_buffer("dbias", tensor(0.0))

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x * x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel() / nc)
        mom1 = 1 - (1 - self.mom) / math.sqrt(bs - 1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias * (1 - self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step < 100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c = c / self.dbias
        means = sums / c
        vars = (sqrs / c).sub_(means * means)
        if bool(self.batch < 20):
            vars.clamp_min_(0.01)
        x = (x - means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)


def lsuv_module(learn, m, xb):
    """
    LSUV initialization
    """
    mdl = learn.model.cuda()
    h = Hook(m, append_stat)
    while mdl(xb) is not None and abs(h.mean) > 1e-3:
        m.bias -= h.mean
    while mdl(xb) is not None and abs(h.std - 1) > 1e-3:
        m.weight.data /= h.std
    h.remove()
    return h.mean, h.std


def get_batch(dl, learn):
    learn.xb, learn.yb = next(iter(dl))
    learn.do_begin_fit(1)
    learn("begin_batch")
    learn("after_fit")
    return learn.xb, learn.yb


def model_summary(learn, find_all=False, print_mod=False):
    """
    List layers and their sizes
    """
    xb, yb = get_batch(learn.data.valid_dl, learn)
    mods = (
        find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    )
    f = lambda hook, mod, inp, out: print(
        f"====\n{mod}\n" if print_mod else "", out.shape
    )
    with Hooks(mods, f) as hooks:
        learn.model(xb)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def apply_mod(m, f):
    """
    Apply model
    """
    f(m)
    for l in m.children():
        apply_mod(l, f)


def set_grad(m, b):
    if isinstance(m, (nn.Linear, nn.BatchNorm2d)):
        return
    if hasattr(m, "weight"):
        for p in m.parameters():
            p.requires_grad_(b)


def adapt_model(learn, data, saved_path):
    """
    Transfer learning
    """
    st = torch.load(saved_path)
    m = learn.model
    m.load_state_dict(st)
    cut = next(
        i
        for i, o in enumerate(learn.model.children())
        if isinstance(o, nn.AdaptiveAvgPool2d)
    )
    m_cut = learn.model[:cut]
    xb, yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        m_cut, AdaptiveConcatPool2d(), Flatten(), nn.Linear(ni * 2, data.c_out)
    )
    learn.model = m_new
    learn.model.apply(partial(set_grad, b=False))  # apply_mod


def save_model(learn, name, path="."):
    """
    Save state_dict
    """
    st = learn.model.state_dict()
    mdl_path = path / "models"
    if not Path.exists(mdl_path):
        mdl_path.mkdir()
    torch.save(st, mdl_path / name)
    print(f"Saved at {mdl_path/name}")
    return mdl_path / name


def multiple_runner(dict_run, save=True, save_path=""):
    """
    dict run example format:
    in order = [no of epochs , architecture, data, loss function , learning rate, callbacks, optimizer]
        dict_runner = {
    "xres18":[1, partial(xresnet18, c_out=n_classes)(), data, loss_func, .001, cbfs,opt_func],}
    """
    learn = None
    clear_memory()
    for i in dict_run.keys():
        print(f"Training model: {i}")
        val = dict_run[i]
        #         print(len(val))
        learn = Learner(
            val[1], val[2], val[3], lr=val[4], cb_funcs=val[5], opt_func=val[6]
        )
        learn.fit(val[0])
        if save == True:
            save_model(learn, i, save_path)
