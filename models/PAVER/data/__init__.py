import os
from pathlib import Path

from torch import nn
from munch import Munch
from inspect import getmro
from inflection import underscore
from torch.utils.data import Dataset

try:
    from exp import ex
except ImportError:
    try:
        from models.PAVER.exp import ex
    except:
        from PAVER.exp import ex


dataset_dict = {}


def add_dataset():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != '__init__':
            try:
                __import__(f"{parent}.{name}")
            except:
                try:
                    __import__(f"models.PAVER.{parent}.{name}")
                except:
                    __import__(f"PAVER.{parent}.{name}")
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, '__mro__') and Dataset in getmro(member):
                    if str(member.__name__) == 'Dataset':
                        continue
                    dataset_dict[str(member.__name__)] = member


add_dataset()