"""ProtoTorch initialization functions."""

from itertools import chain

import torch

INITIALIZERS = dict()


def register_initializer(function):
    """Add the initializer to the registry."""
    INITIALIZERS[function.__name__] = function
    return function


def labels_from(distribution, one_hot=True):
    """Takes a distribution tensor and returns a labels tensor."""
    num_classes = distribution.shape[0]
    llist = [[i] * n for i, n in zip(range(num_classes), distribution)]
    # labels = [l for cl in llist for l in cl]  # flatten the list of lists
    flat_llist = list(chain(*llist))  # flatten label list with itertools.chain
    plabels = torch.tensor(flat_llist, requires_grad=False)
    if one_hot:
        return torch.eye(num_classes)[plabels]
    return plabels


@register_initializer
def ones(x_train, y_train, prototype_distribution, one_hot=True):
    num_protos = torch.sum(prototype_distribution)
    protos = torch.ones(num_protos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution, one_hot)
    return protos, plabels


@register_initializer
def zeros(x_train, y_train, prototype_distribution, one_hot=True):
    num_protos = torch.sum(prototype_distribution)
    protos = torch.zeros(num_protos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution, one_hot)
    return protos, plabels


@register_initializer
def rand(x_train, y_train, prototype_distribution, one_hot=True):
    num_protos = torch.sum(prototype_distribution)
    protos = torch.rand(num_protos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution, one_hot)
    return protos, plabels


@register_initializer
def randn(x_train, y_train, prototype_distribution, one_hot=True):
    num_protos = torch.sum(prototype_distribution)
    protos = torch.randn(num_protos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution, one_hot)
    return protos, plabels


@register_initializer
def stratified_mean(x_train, y_train, prototype_distribution, one_hot=True):
    num_protos = torch.sum(prototype_distribution)
    pdim = x_train.shape[1]
    protos = torch.empty(num_protos, pdim)
    plabels = labels_from(prototype_distribution, one_hot)
    for i, label in enumerate(plabels):
        matcher = torch.eq(label.unsqueeze(dim=0), y_train)
        if one_hot:
            num_classes = y_train.size()[1]
            matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
        xl = x_train[matcher]
        mean_xl = torch.mean(xl, dim=0)
        protos[i] = mean_xl
    plabels = labels_from(prototype_distribution, one_hot=one_hot)
    return protos, plabels


@register_initializer
def stratified_random(x_train,
                      y_train,
                      prototype_distribution,
                      one_hot=True,
                      epsilon=1e-7):
    num_protos = torch.sum(prototype_distribution)
    pdim = x_train.shape[1]
    protos = torch.empty(num_protos, pdim)
    plabels = labels_from(prototype_distribution, one_hot)
    for i, label in enumerate(plabels):
        matcher = torch.eq(label.unsqueeze(dim=0), y_train)
        if one_hot:
            num_classes = y_train.size()[1]
            matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
        xl = x_train[matcher]
        rand_index = torch.zeros(1).long().random_(0, xl.shape[0] - 1)
        random_xl = xl[rand_index]
        protos[i] = random_xl + epsilon
    plabels = labels_from(prototype_distribution, one_hot=one_hot)
    return protos, plabels


def get_initializer(funcname):
    """Deserialize the initializer."""
    if callable(funcname):
        return funcname
    if funcname in INITIALIZERS:
        return INITIALIZERS.get(funcname)
    raise NameError(f"Initializer {funcname} was not found.")
