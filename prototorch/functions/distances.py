"""ProtoTorch distance functions."""

import numpy as np
import torch

from prototorch.functions.helper import (
    _check_shapes,
    _int_and_mixed_shape,
    equal_int_shape,
)


def squared_euclidean_distance(x, y):
    r"""Compute the squared Euclidean distance between :math:`\bm x` and :math:`\bm y`.

    Compute :math:`{\langle \bm x - \bm y \rangle}_2`

    :param `torch.tensor` x: Two dimensional vector
    :param `torch.tensor` y: Two dimensional vector

    **Alias:**
    ``prototorch.functions.distances.sed``
    """
    expanded_x = x.unsqueeze(dim=1)
    batchwise_difference = y - expanded_x
    differences_raised = torch.pow(batchwise_difference, 2)
    distances = torch.sum(differences_raised, axis=2)
    return distances


def euclidean_distance(x, y):
    r"""Compute the Euclidean distance between :math:`x` and :math:`y`.

    Compute :math:`\sqrt{{\langle \bm x - \bm y \rangle}_2}`

    :param `torch.tensor` x: Input Tensor of shape :math:`X \times N`
    :param `torch.tensor` y: Input Tensor of shape :math:`Y \times N`

    :returns: Distance Tensor of shape :math:`X \times Y`
    :rtype: `torch.tensor`
    """
    distances_raised = squared_euclidean_distance(x, y)
    distances = torch.sqrt(distances_raised)
    return distances


def euclidean_distance_v2(x, y):
    diff = y - x.unsqueeze(1)
    pairwise_distances = (diff @ diff.permute((0, 2, 1))).sqrt()
    # Passing `dim1=-2` and `dim2=-1` to `diagonal()` takes the
    # batch diagonal. See:
    # https://pytorch.org/docs/stable/generated/torch.diagonal.html
    distances = torch.diagonal(pairwise_distances, dim1=-2, dim2=-1)
    # print(f"{diff.shape=}")  # (nx, ny, ndim)
    # print(f"{pairwise_distances.shape=}")  # (nx, ny, ny)
    # print(f"{distances.shape=}")  # (nx, ny)
    return distances


def lpnorm_distance(x, y, p):
    r"""Calculate the lp-norm between :math:`\bm x` and :math:`\bm y`.
    Also known as Minkowski distance.

    Compute :math:`{\| \bm x - \bm y \|}_p`.

    Calls ``torch.cdist``

    :param `torch.tensor` x: Two dimensional vector
    :param `torch.tensor` y: Two dimensional vector
    :param p: p parameter of the lp norm
    """
    distances = torch.cdist(x, y, p=p)
    return distances


def omega_distance(x, y, omega):
    r"""Omega distance.

    Compute :math:`{\| \Omega \bm x - \Omega \bm y \|}_p`

    :param `torch.tensor` x: Two dimensional vector
    :param `torch.tensor` y: Two dimensional vector
    :param `torch.tensor` omega: Two dimensional matrix
    """
    projected_x = x @ omega
    projected_y = y @ omega
    distances = squared_euclidean_distance(projected_x, projected_y)
    return distances


def lomega_distance(x, y, omegas):
    r"""Localized Omega distance.

    Compute :math:`{\| \Omega_k \bm x - \Omega_k \bm y_k \|}_p`

    :param `torch.tensor` x: Two dimensional vector
    :param `torch.tensor` y: Two dimensional vector
    :param `torch.tensor` omegas: Three dimensional matrix
    """
    projected_x = x @ omegas
    projected_y = torch.diagonal(y @ omegas).T
    expanded_y = torch.unsqueeze(projected_y, dim=1)
    batchwise_difference = expanded_y - projected_x
    differences_squared = batchwise_difference**2
    distances = torch.sum(differences_squared, dim=2)
    distances = distances.permute(1, 0)
    return distances


def euclidean_distance_matrix(x, y, squared=False, epsilon=1e-10):
    r"""Computes an euclidean distances matrix given two distinct vectors.
    last dimension must be the vector dimension!
    compute the distance via the identity of the dot product. This avoids the memory overhead due to the subtraction!

    - ``x.shape = (number_of_x_vectors, vector_dim)``
    - ``y.shape = (number_of_y_vectors, vector_dim)``

    output: matrix of distances (number_of_x_vectors, number_of_y_vectors)
    """
    for tensor in [x, y]:
        if tensor.ndim != 2:
            raise ValueError(
                "The tensor dimension must be two. You provide: tensor.ndim=" +
                str(tensor.ndim) + ".")
    if not equal_int_shape([tuple(x.shape)[1]], [tuple(y.shape)[1]]):
        raise ValueError(
            "The vector shape must be equivalent in both tensors. You provide: tuple(y.shape)[1]="
            + str(tuple(x.shape)[1]) + " and  tuple(y.shape)(y)[1]=" +
            str(tuple(y.shape)[1]) + ".")

    y = torch.transpose(y)

    diss = (torch.sum(x**2, axis=1, keepdims=True) - 2 * torch.dot(x, y) +
            torch.sum(y**2, axis=0, keepdims=True))

    if not squared:
        if epsilon == 0:
            diss = torch.sqrt(diss)
        else:
            diss = torch.sqrt(torch.max(diss, epsilon))

    return diss


def tangent_distance(signals, protos, subspaces, squared=False, epsilon=1e-10):
    r"""Tangent distances based on the tensorflow implementation of Sascha Saralajews

    For more info about Tangen distances see

    DOI:10.1109/IJCNN.2016.7727534.

    The subspaces is always assumed as transposed and must be orthogonal!
    For local non sparse signals subspaces must be provided!

    - shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    - shape(protos): proto_number x dim1 x dim2 x ... x dimN
    - shape(subspaces): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)

    subspace should be orthogonalized
    Pytorch implementation of Sascha Saralajew's tensorflow code.
    Translation by Christoph Raab
    """
    signal_shape, signal_int_shape = _int_and_mixed_shape(signals)
    proto_shape, proto_int_shape = _int_and_mixed_shape(protos)
    subspace_int_shape = tuple(subspaces.shape)

    # check if the shapes are correct
    _check_shapes(signal_int_shape, proto_int_shape)

    atom_axes = list(range(3, len(signal_int_shape)))
    # for sparse signals, we use the memory efficient implementation
    if signal_int_shape[1] == 1:
        signals = torch.reshape(signals, [-1, np.prod(signal_shape[3:])])

        if len(atom_axes) > 1:
            protos = torch.reshape(protos, [proto_shape[0], -1])

        if subspaces.ndim == 2:
            # clean solution without map if the matrix_scope is global
            projectors = torch.eye(subspace_int_shape[-2]) - torch.dot(
                subspaces, torch.transpose(subspaces))

            projected_signals = torch.dot(signals, projectors)
            projected_protos = torch.dot(protos, projectors)

            diss = euclidean_distance_matrix(projected_signals,
                                             projected_protos,
                                             squared=squared,
                                             epsilon=epsilon)

            diss = torch.reshape(
                diss, [signal_shape[0], signal_shape[2], proto_shape[0]])

            return torch.permute(diss, [0, 2, 1])

        else:

            # no solution without map possible --> memory efficient but slow!
            projectors = torch.eye(subspace_int_shape[-2]) - torch.bmm(
                subspaces,
                subspaces)  # K.batch_dot(subspaces, subspaces, [2, 2])

            projected_protos = (protos @ subspaces
                                ).T  # K.batch_dot(projectors, protos, [1, 1]))

            def projected_norm(projector):
                return torch.sum(torch.dot(signals, projector)**2, axis=1)

            diss = (torch.transpose(map(projected_norm, projectors)) -
                    2 * torch.dot(signals, projected_protos) +
                    torch.sum(projected_protos**2, axis=0, keepdims=True))

            if not squared:
                if epsilon == 0:
                    diss = torch.sqrt(diss)
                else:
                    diss = torch.sqrt(torch.max(diss, epsilon))

            diss = torch.reshape(
                diss, [signal_shape[0], signal_shape[2], proto_shape[0]])

            return torch.permute(diss, [0, 2, 1])

    else:
        signals = signals.permute([0, 2, 1] + atom_axes)

        diff = signals - protos

        # global tangent space
        if subspaces.ndim == 2:
            # Scope Projectors
            projectors = subspaces  #

            # Scope: Tangentspace Projections
            diff = torch.reshape(
                diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
            projected_diff = diff @ projectors
            projected_diff = torch.reshape(
                projected_diff,
                (signal_shape[0], signal_shape[2], signal_shape[1]) +
                signal_shape[3:],
            )

            diss = torch.norm(projected_diff, 2, dim=-1)
            return diss.permute([0, 2, 1])

        # local tangent spaces
        else:
            # Scope: Calculate Projectors
            projectors = subspaces

            # Scope: Tangentspace Projections
            diff = torch.reshape(
                diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
            diff = diff.permute([1, 0, 2])
            projected_diff = torch.bmm(diff, projectors)
            projected_diff = torch.reshape(
                projected_diff,
                (signal_shape[1], signal_shape[0], signal_shape[2]) +
                signal_shape[3:],
            )

            diss = torch.norm(projected_diff, 2, dim=-1)
            return diss.permute([1, 0, 2]).squeeze(-1)


class KernelDistance:
    r"""Kernel Distance

    Distance based on a kernel function.
    """
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    def __call__(self, x_batch, y_batch):
        remove_dims = 0
        # Extend Single inputs
        if len(x_batch.shape) == 1:
            x_batch = [x_batch]
            remove_dims += 1
        if len(y_batch.shape) == 1:
            y_batch = [y_batch]
            remove_dims += 1

        # Loop over batches
        output = []
        for x in x_batch:
            output.append([])
            for y in y_batch:
                output[-1].append(self.single_call(x, y))

        output = torch.Tensor(output)
        for _ in range(remove_dims):
            output.squeeze_(0)

        return output

    def single_call(self, x, y):
        kappa_xx = self.kernel_fn(x, x)
        kappa_xy = self.kernel_fn(x, y)
        kappa_yy = self.kernel_fn(y, y)

        squared_distance = kappa_xx - 2 * kappa_xy + kappa_yy

        return torch.sqrt(squared_distance)


class SquaredKernelDistance(KernelDistance):
    r"""Squared Kernel Distance

    Kernel distance without final squareroot.
    """
    def single_call(self, x, y):
        kappa_xx = self.kernel_fn(x, x)
        kappa_xy = self.kernel_fn(x, y)
        kappa_yy = self.kernel_fn(y, y)

        return kappa_xx - 2 * kappa_xy + kappa_yy


# Aliases
sed = squared_euclidean_distance
