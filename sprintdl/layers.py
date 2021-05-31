from enum import Enum
from pathlib import Path

import torch
import torchvision.models as m
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from .core import *

# Helper Layers


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


class ResizeBatch(nn.Module):
    def __init__(self, *size):
        self.size = size

    def forward(self, x):
        return x.view((x.size(0),) + self.size)


def conv1d_spectral(ni, no, ks=1, stride=1, padding=0, bias=False):
    """
    Conv1d with spectral norm
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return spectral_norm(conv)


class PooledSelfAttention2d(nn.Module):
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.theta = spectral_norm(conv2d(n_channels, n_channels // 8, 1))  # query
        self.phi = spectral_norm(conv2d(n_channels, n_channels // 8, 1))  # key
        self.g = spectral_norm(conv2d(n_channels, n_channels // 2, 1))  # key
        self.o = spectral_norm(conv2d(n_channels // 2, n_channels, 1))  # key
        self.gamma = nn.Parameter(tensor([0.0]))

    def forward(self, x):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3])

        phi = phi.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.n_channels // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.tranpose(1, 2), phi), -1)
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(
                -1, self.n_channels / 2, x.shape[2], x.shape[3]
            )
        )
        return self.gamma * o + x


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        self.query = conv1d_spectral(n_channels, n_channels // 8)
        self.key = conv1d_spectral(n_channels, n_channels // 8)
        self.value = conv1d_spectral(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.0]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


NormType = Enum("NormType", "Batch BatchZero Weight Spectral")


class ShortcutLayer(nn.Module):
    def __init__(self, dense=False):
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


class PartialLayer(nn.Module):
    def __init__(self, func, **kwargs):
        self.repr, self.func = f"{func}({kwargs})", partial(func, **kwargs)

        def forward(self, x):
            return self.func(x)

        def __repr__(self):
            return self.repr


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, 2])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class PixelShuffleICNR(nn.Module):
    def __init__(
        self, ni, nf, scale=2, blur=False, norm_type=NormType.Weight, leaky=None
    ):
        nf = ifnone(nf, ni)
        self.conv = conv_layer(
            ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = GeneralRelu(leak=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


# Helper functions


def relu(inplace: bool = False, leaky: float = None):
    return (
        nn.LeakyReLU(inplace=inplace, negative_slope=leaky)
        if leaky is not None
        else nn.ReLU(inplace=inplace)
    )


def batchnorm_2d(nf, norm_type):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0.0 if norm_type == NormType.BatchZero else 1.0)
    return bn


def ifnone(a, b):
    return b if a is None else a


def model_summary(learn, find_all=False, print_mod=False):
    """
    List layers and their sizes
    """
    xb, yb = get_batch(learn.data.valid_dl, learn)
    mods = (
        find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    )

    def f(hook, mod, inp, out):
        return print(f"====\n{mod}\n" if print_mod else "", out.shape)

    with Hooks(mods, f) as hooks:
        learn.model(xb)


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        self.val = p

    def forward(self, x):
        return x


def children_and_parameters(m: nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children


flatten_model = (
    lambda m: sum(map(flatten_model, children_and_parameters(m)), [])
    if num_children(m)
    else [m]
)


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


def noop(x):
    return x


def get_vision_model(name, n_classes, pretrained=False):
    try:
        return getattr(m, name)(num_classes=n_classes, pretrained=pretrained)
    except:
        return getattr(m, name)(num_classes=n_classes)


def param_state(x):
    return x.requires_grad


def total_layer_state(learn):
    ps = [param_state(x) for x in learn.model.parameters()]
    frozen = ps.count(False)
    return f"Frozen: {frozen}, Not: {len(ps)-frozen}, Total: {len(ps)}"


class FreezeUnfreeze:
    def __init__(self, learn, switch, to=None):
        self.model = learn.model
        self.switch = switch  # 0 for freeze, 1 for unfreeze
        self.ps = [None for x in learn.model.parameters()]
        self.count = 0
        self.to = to

    def runner(self):
        if self.to == None:
            self.to = len(self.ps)
        if self.to < 0:
            self.to = len(self.ps) - abs(self.to)
        for param in self.model.parameters():
            if self.count < self.to:
                param.requires_grad = False if self.switch == 0 else True
                self.count += 1


def freeze_to(learn, to=None):
    FreezeUnfreeze(learn, 0, to).runner()


def unfreeze_to(learn, to=None):
    FreezeUnfreeze(learn, 1, to).runner()


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


# Generalized Versions


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


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


# Inits


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> nn.Module:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, "weight"):
            func(m.weight)
        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    return m


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


# for learning only


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


class SequentialEx(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, l):
        return self.layers.append(l)

    def extend(self, l):
        return self.layers.extend(l)

    def insert(self, i, l):
        return self.layers.insert(i, l)
