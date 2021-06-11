"""ProtoFlow utilities"""

import numpy as np


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
