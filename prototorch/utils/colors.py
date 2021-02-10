"""ProtoFlow color utilities."""

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
import matplotlib.lines as mlines


def color_scheme(n, cmap="viridis", form="hex", tikz=False,
                 zero_indexed=False):
    """Return *n* colors from the color scheme.

    Arguments:
        n (int): number of colors to return

    Keyword Arguments:
        cmap (str): Name of a matplotlib `colormap\
            <https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html>`_.
        form (str): Colorformat (supports "hex" and "rgb").
        tikz (bool): Output as `TikZ <https://github.com/pgf-tikz/pgf>`_
            command.
        zero_indexed (bool): Use zero indexing for output array.

    Returns:
        (list): List of colors
    """
    cmap = cm.get_cmap(cmap)
    colornorm = Normalize(vmin=1, vmax=n)
    hex_map = dict()
    rgb_map = dict()
    for cl in range(1, n + 1):
        if zero_indexed:
            hex_map[cl - 1] = to_hex(cmap(colornorm(cl)))
            rgb_map[cl - 1] = to_rgb(cmap(colornorm(cl)))
        else:
            hex_map[cl] = to_hex(cmap(colornorm(cl)))
            rgb_map[cl] = to_rgb(cmap(colornorm(cl)))
    if tikz:
        for k, v in rgb_map.items():
            print(f"\\definecolor{{color-{k}}}{{rgb}}{{{v[0]},{v[1]},{v[2]}}}")
    if form == "hex":
        return hex_map
    elif form == "rgb":
        return rgb_map
    else:
        return hex_map


def get_legend_handles(labels, marker="dots", zero_indexed=False):
    """Return matplotlib legend handles and colors."""
    handles = list()
    n = len(labels)
    colors = color_scheme(n,
                          cmap="viridis",
                          form="hex",
                          zero_indexed=zero_indexed)
    for label, color in zip(labels, colors.values()):
        if marker == "dots":
            handle = mlines.Line2D([], [],
                                   color="white",
                                   markerfacecolor=color,
                                   marker="o",
                                   markersize=10,
                                   markeredgecolor="k",
                                   label=label)
        else:
            handle = mlines.Line2D([], [],
                                   color=color,
                                   marker="",
                                   markersize=15,
                                   label=label)
            handles.append(handle)
    return handles, colors
