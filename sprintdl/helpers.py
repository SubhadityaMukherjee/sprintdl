from collections import OrderedDict
from pathlib import Path
from typing import *

import torch
from fastprogress.fastprogress import progress_bar
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms


def normalize(x, m, s):
    return (x - m) / s


def listify(o):
    """
    Convert to list
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def normalize_to(train, valid):
    """
    To a specific mean and std
    """
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)


def flatten(x):
    """
    Flatten tensor
    """
    return x.view(x.shape[0], -1)


def mnist_resize(x):
    """
    To mnist size
    """
    return x.view(-1, 1, 28, 28)


def get_hist(h):
    """
    grab histogram
    """
    return torch.stack(h.stats[2]).t().float().log1p()


def get_min(h):
    """
    Grab min values from histogram
    """
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[19:22].sum(0) / h1.sum(0)


class ListContainer:
    """
    Container of list items
    """

    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx) == len(self)  # bool mask
            return [o for m, o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    # export
    # export

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


def get_batch(dl):
    """
    Grab one batch
    """
    dl.xb, dl.yb = next(iter(dl))
    for cb in dl.cbs:
        cb.set_runner(dl)
    dl("begin_batch")
    return dl.xb, dl.yb


def find_modules(m, cond):
    """
    Return modules with a condition
    """
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    """
    Check if linear
    """
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(l, lin_layers)


def append_stat(hook, mod, inp, outp):
    """
    Add to stats mean std
    """
    d = outp.data
    hook.mean, hook.std = d.mean().item(), d.std().item()


def setify(i):
    """
    Convert to set
    """
    return i if isinstance(i, set) else set(listify(i))


def compose(x, funcs, *args, order_key="_order", **kwargs):
    """
    Chain functions
    """
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def uniqueify(x, sort=False):
    """
    Get unique dictionary
    """
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res


def show_image(im, figsize=(3, 3)):
    """
    Show single image
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(im.permute(1, 2, 0))


def timeit(method):
    """
    Helper to time a function
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def lin_comb(v1, v2, beta):
    """
    Linear Combination
    """
    return beta * v1 + (1 - beta) * v2


def param_getter(m):
    """
    Grab params
    """
    return m.parameters()


def unsqueeze(input, dims):
    """
    Add a dim
    """
    for dim in listify(dims):
        input = torch.unsqueeze(input, dim)
        return input


def clear_memory():
    """
    Clear GPU cache
    """
    torch.cuda.empty_cache()


def image_loader(image_name, imsize=256):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.cuda()


def get_class_pred(im_path, learn, ll, imsize=256):
    """
    Get a prediction for classification for single image
    """
    temp = Path(im_path)
    learn.model.eval()
    preds = learn.model(image_loader(temp, imsize))
    preds = int(preds.max(1, keepdim=True)[1].detach())
    lab = dict(map(reversed, ll.train.proc_y.otoi.items()))[preds]
    return lab


def get_label_dict(ll):
    """
    Get label dictionary
    """
    return dict(map(reversed, ll.train.proc_y.otoi.items()))


def classification_report(learn, n_classes, device):
    """
    Confusion matrix
    """
    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for i, (inputs, classes) in progress_bar(
            enumerate(learn.data.valid_dl), total=len(learn.data.valid_dl)
        ):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = learn.model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    print(confusion_matrix)
