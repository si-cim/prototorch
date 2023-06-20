"""ProtoTorch distances"""

import torch


def squared_euclidean_distance(x, y):
    r"""Compute the squared Euclidean distance between :math:`\bm x` and :math:`\bm y`.

    Compute :math:`{\langle \bm x - \bm y \rangle}_2`

    **Alias:**
    ``prototorch.functions.distances.sed``
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    expanded_x = x.unsqueeze(dim=1)
    batchwise_difference = y - expanded_x
    differences_raised = torch.pow(batchwise_difference, 2)
    distances = torch.sum(differences_raised, axis=2)
    return distances


def euclidean_distance(x, y):
    r"""Compute the Euclidean distance between :math:`x` and :math:`y`.

    Compute :math:`\sqrt{{\langle \bm x - \bm y \rangle}_2}`

    :returns: Distance Tensor of shape :math:`X \times Y`
    :rtype: `torch.tensor`
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    distances_raised = squared_euclidean_distance(x, y)
    distances = torch.sqrt(distances_raised)
    return distances


def euclidean_distance_v2(x, y):
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    diff = y - x.unsqueeze(1)
    pairwise_distances = (diff @ diff.permute((0, 2, 1))).sqrt()
    # Passing `dim1=-2` and `dim2=-1` to `diagonal()` takes the
    # batch diagonal. See:
    # https://pytorch.org/docs/stable/generated/torch.diagonal.html
    distances = torch.diagonal(pairwise_distances, dim1=-2, dim2=-1)
    return distances


def lpnorm_distance(x, y, p):
    r"""Calculate the lp-norm between :math:`\bm x` and :math:`\bm y`.
    Also known as Minkowski distance.

    Compute :math:`{\| \bm x - \bm y \|}_p`.

    Calls ``torch.cdist``

    :param p: p parameter of the lp norm
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    distances = torch.cdist(x, y, p=p)
    return distances


def omega_distance(x, y, omega):
    r"""Omega distance.

    Compute :math:`{\| \Omega \bm x - \Omega \bm y \|}_p`

    :param `torch.tensor` omega: Two dimensional matrix
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    projected_x = x @ omega
    projected_y = y @ omega
    distances = squared_euclidean_distance(projected_x, projected_y)
    return distances


def lomega_distance(x, y, omegas):
    r"""Localized Omega distance.

    Compute :math:`{\| \Omega_k \bm x - \Omega_k \bm y_k \|}_p`

    :param `torch.tensor` omegas: Three dimensional matrix
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    projected_x = x @ omegas
    projected_y = torch.diagonal(y @ omegas).T
    expanded_y = torch.unsqueeze(projected_y, dim=1)
    batchwise_difference = expanded_y - projected_x
    differences_squared = batchwise_difference**2
    distances = torch.sum(differences_squared, dim=2)
    distances = distances.permute(1, 0)
    return distances


# Aliases
sed = squared_euclidean_distance
