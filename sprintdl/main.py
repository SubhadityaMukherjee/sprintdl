import warnings

import torch

from .aug import *
from .callbacks import *
from .core import *
from .data import *
from .helpers import *
from .layers import *
from .loss import *
from .models import *
from .optimizers import *
from .tests import *

warnings.filterwarnings("ignore", category=UserWarning)


torch.Tensor.ndim = property(lambda x: len(x.shape))
