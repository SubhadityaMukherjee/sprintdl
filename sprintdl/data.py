import mimetypes
import os
import random
import re
import shutil
import sys
import urllib
from functools import partial
from pathlib import Path

import pandas as pd
import PIL
import torch
import torchvision.datasets as d
import wget

from .core import Learner, get_dls
from .helpers import *
from .layers import save_model

Path.ls = lambda x: list(x.iterdir())
"""
This part contains the defintions for everything needed to work with data
"""


def get_vision_datasets(fpath, name):
    return getattr(d, name)(root=fpath, download=True)


def untar_data(fpath):
    """
    Extract data using shutil
    """
    fin_name = Path.joinpath(fpath.parent, fpath.stem)
    if not Path.exists(fin_name):
        shutil.unpack_archive(fpath, fin_name)
    print(f"Extracted to {fin_name}")
    return fin_name


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)

AUDIO_EXTS = {
    str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith("audio/")
}


def get_name_from_url(url, name):
    """
    Take a url and grab the name if not specified
    """
    if len(name) != None:
        return url.split("/")[-1]
    else:
        return name


def bar_progress(current, total, width=80):
    """
    Custom progress bar for dataset download
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_and_check(url, fpath=".", name=""):
    """
    Download and save if url or just take path
    """
    down_path = Path.joinpath(Path(fpath), get_name_from_url(url, name))
    if any(x for x in ["www", "http"] if x in url):
        status_code = urllib.request.urlopen(url).getcode()
        if status_code != 200:
            print("Sorry, invalid url")
        else:
            if not Path.exists(down_path):
                wget.download(url, str(down_path), bar=bar_progress)
                print(f"Downloaded to {down_path}")
            else:
                print(f"Already downloaded at {down_path}")
    else:
        if not Path.exists(down_path):
            return "Invalid path"
    return down_path


class Dataset:
    """
    Return x,y
    """

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ## Sampler


class Sampler:
    """
    Shuffle, randperm
    """

    def __init__(self, ds, bs, shuffle=False):
        self.n, self.bs, self.shuffle = len(ds), bs, shuffle

    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            yield self.idxs[i : i + self.bs]


def collate(b):
    """
    Stackers
    """
    xs, ys = zip(*b)
    return torch.stack(xs), torch.stack(ys)


class DataLoader:
    """
    Load -> sample -> collate
    """

    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds, self.sampler, self.collate_fn = ds, sampler, collate_fn

    def __iter__(self):
        for s in self.sampler:
            yield self.collate_fn([self.ds[i] for i in s])


class DataBunch:
    """
    Group of train,valid
    """

    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl, self.valid_dl, self.c_in, self.c_out = (
            train_dl,
            valid_dl,
            c_in,
            c_out,
        )

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def _get_files(p, fs, extensions=None):
    """
    Ignores dot files and returns with extensions
    """
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    """
    List files
    """
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)


class ItemList(ListContainer):
    """
    Item class
    """

    def __init__(self, items, path=".", tfms=None):
        super().__init__(items)
        super(SplitData).__init__()
        self.path, self.tfms = Path(path), tfms

    def __repr__(self):
        return f"{super().__repr__()}\nPath: {self.path}"

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, tfms=self.tfms)

    def get(self, i):
        return i

    def _get(self, i):
        return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, list):
            return [self._get(o) for o in res]
        return self._get(res)


class ImageList(ItemList):
    """
    Only lists images
    """

    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None:
            extensions = image_extensions
        return cls(
            get_files(path, extensions, recurse=recurse, include=include),
            path,
            **kwargs,
        )

    def get(self, fn):
        return PIL.Image.open(fn)


def grandparent_splitter(fn, valid_name="valid", train_name="train"):
    """
    Grab grandparent from path
    """
    gp = fn.parent.parent.name
    return True if gp == valid_name else False if gp == train_name else None


def split_by_func(items, f):
    mask = [f(o) for o in items]
    f = [o for o, m in zip(items, mask) if m == False]
    t = [o for o, m in zip(items, mask) if m == True]
    return f, t


class SplitData:
    def __init__(self, train, valid):
        self.train, self.valid = train, valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    def __setstate__(self, data: Any):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, il, f):
        lists = map(il.new, split_by_func(il.items, f))
        return cls(*lists)

    def __repr__(self):
        return f"{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n"


def databunchify(sd, bs, c_in=None, c_out=None, **kwargs):
    """
    Convert to databunch
    """
    dls = get_dls(sd.train, sd.valid, bs, **kwargs)
    return DataBunch(*dls, c_in=c_in, c_out=c_out)


SplitData.to_databunch = databunchify


class Processor:
    """
    To apply labels
    """

    def process(self, items):
        return items


class CategoryProcessor(Processor):
    """
    Grab categories
    """

    def __init__(self):
        self.vocab = None

    def __call__(self, items):
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {v: k for k, v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]

    def proc1(self, item):
        return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]

    def deproc1(self, idx):
        return self.vocab[idx]


def TableLoader(
    fname,
    path_col,
    target_col,
    add_before="",
    add_after="",
    file_type="csv",
    custom_read=None,
):
    if custom_read != None:
        df = custom_read
    else:
        df = getattr(pd, f"read_{file_type}")(fname)
    if len(add_before) > 1:
        df[path_col] = df[path_col].apply(lambda x: add_before + x)
    if len(add_after) > 1:
        df[path_col] = df[path_col].apply(lambda x: x + add_after)
    df[path_col] = df[path_col].apply(lambda x: Path(x))
    return dict(zip(df[path_col].values, df[target_col].values))


def parent_labeler(fn):
    """
    Return path parent
    """
    return fn.parent.name


def table_labeler(fn, dic):
    """
    Return label from dict
    """
    return dic[fn]


def _label_by_func(ds, f, cls=ItemList):
    """
    Label using a custom fnction
    """
    return cls([f(o) for o in ds.items], path=ds.path)


def re_labeler(fn, pat):
    return re.findall(pat, str(fn))[0]


class LabeledData:
    def process(self, il, procs):
        return il.new(compose(il.items, procs))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x, self.y = self.process(x, proc_x), self.process(y, proc_y)
        self.proc_x, self.proc_y = proc_x, proc_y

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny:{self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx):
        return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (
            isinstance(idx, torch.LongTensor) and not idx.ndim
        )
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)


def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train, valid)


def random_splitter(fn, p_valid):
    """
    Randomly split
    """
    return random.random() < p_valid
