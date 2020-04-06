"""ProtoTorch initialization functions."""

from itertools import chain

import torch

INITIALIZERS = dict()


def register_initializer(func):
    INITIALIZERS[func.__name__] = func
    return func


def labels_from(distribution):
    """Takes a distribution tensor and returns a labels tensor."""
    nclasses = distribution.shape[0]
    llist = [[i] * n for i, n in zip(range(nclasses), distribution)]
    # labels = [l for cl in llist for l in cl]  # flatten the list of lists
    labels = list(chain(*llist))  # flatten using itertools.chain
    return torch.tensor(labels, requires_grad=False)


@register_initializer
def ones(x_train, y_train, prototype_distribution):
    nprotos = torch.sum(prototype_distribution)
    protos = torch.ones(nprotos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution)
    return protos, plabels


@register_initializer
def zeros(x_train, y_train, prototype_distribution):
    nprotos = torch.sum(prototype_distribution)
    protos = torch.zeros(nprotos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution)
    return protos, plabels


@register_initializer
def rand(x_train, y_train, prototype_distribution):
    nprotos = torch.sum(prototype_distribution)
    protos = torch.rand(nprotos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution)
    return protos, plabels


@register_initializer
def randn(x_train, y_train, prototype_distribution):
    nprotos = torch.sum(prototype_distribution)
    protos = torch.randn(nprotos, *x_train.shape[1:])
    plabels = labels_from(prototype_distribution)
    return protos, plabels


@register_initializer
def stratified_mean(x_train, y_train, prototype_distribution):
    nprotos = torch.sum(prototype_distribution)
    pdim = x_train.shape[1]
    protos = torch.empty(nprotos, pdim)
    plabels = labels_from(prototype_distribution)
    for i, l in enumerate(plabels):
        xl = x_train[y_train == l]
        mean_xl = torch.mean(xl, dim=0)
        protos[i] = mean_xl
    return protos, plabels


@register_initializer
def stratified_random(x_train, y_train, prototype_distribution):
    gen = torch.manual_seed(torch.initial_seed())
    nprotos = torch.sum(prototype_distribution)
    pdim = x_train.shape[1]
    protos = torch.empty(nprotos, pdim)
    plabels = labels_from(prototype_distribution)
    for i, l in enumerate(plabels):
        xl = x_train[y_train == l]
        rand_index = torch.zeros(1).long().random_(0,
                                                   xl.shape[1] - 1,
                                                   generator=gen)
        random_xl = xl[rand_index]
        protos[i] = random_xl
    return protos, plabels


def get_initializer(funcname):
    if callable(funcname):
        return funcname
    else:
        if funcname in INITIALIZERS:
            return INITIALIZERS.get(funcname)
        else:
            raise NameError(f'Initializer {funcname} was not found.')
