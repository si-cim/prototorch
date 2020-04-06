"""ProtoTorch distance functions."""

import torch


def squared_euclidean_distance(x, y):
    """Compute the squared Euclidean distance between :math:`x` and :math:`y`.

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    """
    expanded_x = x.unsqueeze(dim=1)
    batchwise_difference = y - expanded_x
    differences_raised = torch.pow(batchwise_difference, 2)
    distances = torch.sum(differences_raised, axis=2)
    return distances


def euclidean_distance(x, y):
    """Compute the Euclidean distance between :math:`x` and :math:`y`.

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    """
    distances_raised = squared_euclidean_distance(x, y)
    distances = torch.sqrt(distances_raised)
    return distances


def lpnorm_distance(x, y, p):
    """Compute :math:`{\\langle x, y \\rangle}_p`.

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    """
    # # DEPRECATED in favor of torch.cdist
    # expanded_x = x.unsqueeze(dim=1)
    # batchwise_difference = y - expanded_x
    # differences_raised = torch.pow(batchwise_difference, p)
    # distances_raised = torch.sum(differences_raised, axis=2)
    # distances = torch.pow(distances_raised, 1.0 / p)
    # return distances
    distances = torch.cdist(x, y, p=p)
    return distances


def omega_distance(x, y, omega):
    """Omega distance.

    Compute :math:`{\\langle \\Omega x, \\Omega y \\rangle}_p`

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    Expected dimension of omega is 2.
    """
    projected_x = x @ omega
    projected_y = y @ omega
    distances = squared_euclidean_distance(projected_x, projected_y)
    return distances


def lomega_distance(x, y, omegas):
    """Localized Omega distance.

    Compute :math:`{\\langle \\Omega_k x, \\Omega_k y_k \\rangle}_p`

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    Expected dimension of omegas is 3.
    """
    projected_x = x @ omegas
    projected_y = torch.diagonal(y @ omegas).T
    expanded_y = torch.unsqueeze(projected_y, dim=1)
    batchwise_difference = expanded_y - projected_x
    differences_squared = batchwise_difference**2
    distances = torch.sum(differences_squared, dim=2)
    distances = distances.permute(1, 0)
    return distances
