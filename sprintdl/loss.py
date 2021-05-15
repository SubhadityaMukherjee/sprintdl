import torch
from torch import nn

from .aug import reduce_loss
from .core import *


class Mse(Module):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2 * (inp.squeeze() - targ).unsqueeze(-1) / targ.shape[0]


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x - m[:, None]).exp().sum(-1).log()


def log_softmax(x):
    return x - x.logsumexp(-1, keepdim=True)


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss / c, nll, self.ε)
