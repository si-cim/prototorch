"""ProtoTorch activation functions."""

import torch

ACTIVATIONS = dict()


# def register_activation(scriptf):
#     ACTIVATIONS[scriptf.name] = scriptf
#     return scriptf
def register_activation(function):
    """Add the activation function to the registry."""
    ACTIVATIONS[function.__name__] = function
    return function


@register_activation
# @torch.jit.script
def identity(x, beta=0.0):
    """Identity activation function.

    Definition:
    :math:`f(x) = x`

    Keyword Arguments:
        beta (`float`): Ignored.
    """
    return x


@register_activation
# @torch.jit.script
def sigmoid_beta(x, beta=10.0):
    r"""Sigmoid activation function with scaling.

    Definition:
    :math:`f(x) = \frac{1}{1 + e^{-\beta x}}`

    Keyword Arguments:
        beta (`float`): Scaling parameter :math:`\beta`
    """
    out = 1.0 / (1.0 + torch.exp(-1.0 * beta * x))
    return out


@register_activation
# @torch.jit.script
def swish_beta(x, beta=10.0):
    r"""Swish activation function with scaling.

    Definition:
    :math:`f(x) = \frac{x}{1 + e^{-\beta x}}`

    Keyword Arguments:
        beta (`float`): Scaling parameter :math:`\beta`
    """
    out = x * sigmoid_beta(x, beta=beta)
    return out


def get_activation(funcname):
    """Deserialize the activation function."""
    if callable(funcname):
        return funcname
    if funcname in ACTIVATIONS:
        return ACTIVATIONS.get(funcname)
    raise NameError(f"Activation {funcname} was not found.")
