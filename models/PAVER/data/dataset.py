import os
import json
import random
from pathlib import Path
from itertools import chain

import torch
import numpy as np
from munch import Munch
from torch.utils.data import Dataset

try:
    from exp import ex
except ImportError:
    try:
        from models.PAVER.exp import ex
    except:
        from PAVER.exp import ex
from . import dataset_dict


@ex.capture()
def get_dataset(dataset_name, modes=[]):
    outputs = {}
    for mode in modes:
        outputs[mode] = dataset_dict[dataset_name](mode=mode)
    return outputs