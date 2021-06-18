"""ProtoTorch utils test suite"""

import numpy as np
import torch

import prototorch as pt


def test_mesh2d_without_input():
    mesh, xx, yy = pt.utils.mesh2d(border=2.0, resolution=10)
    assert mesh.shape[0] == 100
    assert mesh.shape[1] == 2
    assert xx.shape[0] == 10
    assert xx.shape[1] == 10
    assert yy.shape[0] == 10
    assert yy.shape[1] == 10
    assert np.min(xx) == -2.0
    assert np.max(xx) == 2.0
    assert np.min(yy) == -2.0
    assert np.max(yy) == 2.0


def test_mesh2d_with_torch_input():
    x = 10 * torch.rand(5, 2)
    mesh, xx, yy = pt.utils.mesh2d(x, border=0.0, resolution=100)
    assert mesh.shape[0] == 100 * 100
    assert mesh.shape[1] == 2
    assert xx.shape[0] == 100
    assert xx.shape[1] == 100
    assert yy.shape[0] == 100
    assert yy.shape[1] == 100
    assert np.min(xx) == x[:, 0].min()
    assert np.max(xx) == x[:, 0].max()
    assert np.min(yy) == x[:, 1].min()
    assert np.max(yy) == x[:, 1].max()


def test_hex_to_rgb():
    red_rgb = list(pt.utils.hex_to_rgb(["#ff0000"]))[0]
    assert red_rgb[0] == 255
    assert red_rgb[1] == 0
    assert red_rgb[2] == 0


def test_rgb_to_hex():
    blue_hex = list(pt.utils.rgb_to_hex([(0, 0, 255)]))[0]
    assert blue_hex.lower() == "0000ff"
