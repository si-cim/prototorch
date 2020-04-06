"""ProtoTorch activation functions."""

import torch

ACTIVATIONS = dict()


def register_activation(func):
    ACTIVATIONS[func.__name__] = func
    return func


@register_activation
def identity(input, **kwargs):
    """:math:`f(x) = x`"""
    return input


@register_activation
def sigmoid_beta(input, beta=10):
    """:math:`f(x) = \\frac{1}{1 + e^{-\\beta x}}`

    Keyword Arguments:
        beta (float): Parameter :math:`\\beta`
    """
    out = torch.reciprocal(1.0 + torch.exp(-beta * input))
    return out


@register_activation
def swish_beta(input, beta=10):
    """:math:`f(x) = \\frac{x}{1 + e^{-\\beta x}}`

    Keyword Arguments:
        beta (float): Parameter :math:`\\beta`
    """
    out = input * sigmoid_beta(input, beta=beta)
    return out


def get_activation(funcname):
    if callable(funcname):
        return funcname
    else:
        if funcname in ACTIVATIONS:
            return ACTIVATIONS.get(funcname)
        else:
            raise NameError(f'Activation {funcname} was not found.')
