"""ProtoFlow utilities"""

import warnings
from collections.abc import Iterable
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def mesh2d(x=None, border: float = 1.0, resolution: int = 100):
    if x is not None:
        x_shift = border * np.ptp(x[:, 0])
        y_shift = border * np.ptp(x[:, 1])
        x_min, x_max = x[:, 0].min() - x_shift, x[:, 0].max() + x_shift
        y_min, y_max = x[:, 1].min() - y_shift, x[:, 1].max() + y_shift
    else:
        x_min, x_max = -border, border
        y_min, y_max = -border, border
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    return mesh, xx, yy


def distribution_from_list(list_dist: list[int],
                           clabels: Iterable[int] = None):
    clabels = clabels or list(range(len(list_dist)))
    distribution = dict(zip(clabels, list_dist))
    return distribution


def parse_distribution(user_distribution,
                       clabels: Iterable[int] = None) -> dict[int, int]:
    """Parse user-provided distribution.

    Return a dictionary with integer keys that represent the class labels and
    values that denote the number of components/prototypes with that class
    label.

    The argument `user_distribution` could be any one of a number of allowed
    formats. If it is a Python list, it is assumed that there are as many
    entries in this list as there are classes, and the value at each index of
    this list describes the number of prototypes for that particular class. So,
    [1, 1, 1] implies that we have three classes with one prototype per class.
    If it is a Python tuple, a shorthand of (num_classes, prototypes_per_class)
    is assumed. If it is a Python dictionary, the key-value pairs describe the
    class label and the number of prototypes for that class respectively. So,
    {0: 2, 1: 2, 2: 2} implies that we have three classes with labels {1, 2,
    3}, each equipped with two prototypes. If however, the dictionary contains
    the keys "num_classes" and "per_class", they are parsed to use their values
    as one might expect.

    """
    if isinstance(user_distribution, dict):
        if "num_classes" in user_distribution.keys():
            num_classes = int(user_distribution["num_classes"])
            per_class = int(user_distribution["per_class"])
            return distribution_from_list([per_class] * num_classes, clabels)
        else:
            return user_distribution
    elif isinstance(user_distribution, tuple):
        assert len(user_distribution) == 2
        num_classes, per_class = user_distribution
        num_classes, per_class = int(num_classes), int(per_class)
        return distribution_from_list([per_class] * num_classes, clabels)
    elif isinstance(user_distribution, list):
        return distribution_from_list(user_distribution, clabels)
    else:
        msg = f"`distribution` was not understood." \
            f"You have provided: {user_distribution}."
        raise ValueError(msg)


def parse_data_arg(data_arg: Union[Dataset, DataLoader, list, tuple]):
    """Return data and target as torch tensors."""
    if isinstance(data_arg, Dataset):
        if hasattr(data_arg, "__len__"):
            ds_size = len(data_arg)  # type: ignore
            loader = DataLoader(data_arg, batch_size=ds_size)
            data, targets = next(iter(loader))
        else:
            emsg = f"Dataset {data_arg} is not sized (`__len__` unimplemented)."
            raise TypeError(emsg)

    elif isinstance(data_arg, DataLoader):
        data = torch.tensor([])
        targets = torch.tensor([])
        for x, y in data_arg:
            data = torch.cat([data, x])
            targets = torch.cat([targets, y])
    else:
        assert len(data_arg) == 2
        data, targets = data_arg
        if not isinstance(data, torch.Tensor):
            wmsg = f"Converting data to {torch.Tensor}..."
            warnings.warn(wmsg)
            data = torch.Tensor(data)
        if not isinstance(targets, torch.LongTensor):
            wmsg = f"Converting targets to {torch.LongTensor}..."
            warnings.warn(wmsg)
            targets = torch.LongTensor(targets)
    return data, targets
