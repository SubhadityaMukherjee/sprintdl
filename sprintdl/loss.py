import torch
from torch import nn

from .aug import reduce_loss
from .core import *

# Loss


class Mse(Module):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2 * (inp.squeeze() - targ).unsqueeze(-1) / targ.shape[0]


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


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


# Metrics


def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x - m[:, None]).exp().sum(-1).log()


def log_softmax(x):
    return x - x.logsumexp(-1, keepdim=True)


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return (y_pred > thresh).byte() == y_true.byte().float().mean()


def top_k_accuracy(input, targs, k=5):
    input = input.topk(k=k, dim=-1)[1]
    targs = targs.unsqueeze(dim=-1).expand_as(input)
    return (input == targs).max(dim=-1)[0].float().mean()


def error_rate(input, targs):
    return 1 - accuracy(input, targs)


def flatten_check(out, targ):
    "Check that `out` and `targ` have the same number of elements and flatten them."
    out, targ = out.contiguous().view(-1), targ.contiguous().view(-1)
    assert len(out) == len(
        targ
    ), f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
    return out, targ


def exp_rmspe(pred, targ):
    "Exp RMSE between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred) / targ
    return torch.sqrt((pct_var ** 2).mean())


def mean_absolute_error(pred, targ):
    "Mean absolute error between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    return torch.abs(targ - pred).mean()


def mean_squared_error(pred, targ):
    "Mean squared error between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    return F.mse_loss(pred, targ)


def root_mean_squared_error(pred, targ):
    "Root mean squared error between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    return torch.sqrt(F.mse_loss(pred, targ))


def mean_squared_logarithmic_error(pred, targ):
    "Mean squared logarithmic error between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    return F.mse_loss(torch.log(1 + pred), torch.log(1 + targ))


def psnr(input, targs):
    return 10 * (1.0 / mean_squared_error(input, targs)).log10()


def explained_variance(pred, targ):
    pred, targ = flatten_check(pred, targ)
    var_pct = torch.var(targ - pred) / torch.var(targ)
    return 1 - var_pct


def r2_score(pred, targ):
    pred, targ = flatten_check(pred, targ)
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - u / d


def auc_roc_score(input, targ):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.0).sum(-1)


def roc_curve(input, targ):
    "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
    targ = targ == 1
    desc_score_indices = torch.flip(input.argsort(-1), [-1])
    input = input[desc_score_indices]
    targ = targ[desc_score_indices]
    d = input[1:] - input[:-1]
    distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]
    threshold_idxs = torch.cat(
        (distinct_value_indices, torch.LongTensor([len(targ) - 1]).to(targ.device))
    )
    tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    if tps[0] != 0 or fps[0] != 0:
        zer = fps.new_zeros(1)
        fps = torch.cat((zer, fps))
        tps = torch.cat((zer, tps))
    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
    return fpr, tpr


def dice(input, targs, iou=False, eps=1e-8):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n, -1)
    targs = targs.view(n, -1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input + targs).sum(dim=1).float()
    if not iou:
        l = 2.0 * intersect / union
    else:
        l = intersect / (union - intersect + eps)
    l[union == 0.0] = 1.0
    return l.mean()


def WasserteinLoss(real, fake):
    return real.mean() - fake.mean()


def fbeta(y_pred, y_true, thresh=0.2, beta=2, eps=1e-9, sigmoid=True):
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean()
