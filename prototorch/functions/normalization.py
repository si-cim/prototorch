# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import torch


def orthogonalization(tensors):
    r""" Orthogonalization of a given tensor via polar decomposition.
    """
    u, _, v = torch.svd(tensors, compute_uv=True)
    u_shape = tuple(list(u.shape))
    v_shape = tuple(list(v.shape))

    # reshape to (num x N x M)
    u = torch.reshape(u, (-1, u_shape[-2], u_shape[-1]))
    v = torch.reshape(v, (-1, v_shape[-2], v_shape[-1]))

    out = u @ v.permute([0, 2, 1])

    out = torch.reshape(out, u_shape[:-1] + (v_shape[-2], ))

    return out


def trace_normalization(tensors):
    r""" Trace normalization
    """
    epsilon = torch.tensor([1e-10], dtype=torch.float64)
    # Scope trace_normalization
    constant = torch.trace(tensors)

    if epsilon != 0:
        constant = torch.max(constant, epsilon)

    return tensors / constant
