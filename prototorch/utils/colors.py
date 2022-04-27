"""ProtoTorch color utilities"""

import matplotlib.lines as mlines
import torch
from matplotlib import cm
from matplotlib.colors import (
    Normalize,
    to_hex,
    to_rgb,
)


def hex_to_rgb(hex_values):
    for v in hex_values:
        v = v.lstrip('#')
        lv = len(v)
        c = [int(v[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
        yield c


def rgb_to_hex(rgb_values):
    for v in rgb_values:
        c = "%02x%02x%02x" % tuple(v)
        yield c


def get_colors(vmax, vmin=0, cmap="viridis"):
    cmap = cm.get_cmap(cmap)
    colornorm = Normalize(vmin=vmin, vmax=vmax)
    colors = dict()
    for c in range(vmin, vmax + 1):
        colors[c] = to_hex(cmap(colornorm(c)))
    return colors


def get_legend_handles(colors, labels, marker="dots", zero_indexed=False):
    handles = list()
    for color, label in zip(colors.values(), labels):
        if marker == "dots":
            handle = mlines.Line2D(
                xdata=[],
                ydata=[],
                label=label,
                color="white",
                markerfacecolor=color,
                marker="o",
                markersize=10,
                markeredgecolor="k",
            )
        else:
            handle = mlines.Line2D(
                xdata=[],
                ydata=[],
                label=label,
                color=color,
                marker="",
                markersize=15,
            )
        handles.append(handle)
    return handles
