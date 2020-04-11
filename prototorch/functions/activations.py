"""ProtoTorch activation functions."""

import torch

ACTIVATIONS = dict()


# def register_activation(scriptf):
#     ACTIVATIONS[scriptf.name] = scriptf
#     return scriptf
def register_activation(f):
    ACTIVATIONS[f.__name__] = f
    return f


@register_activation
# @torch.jit.script
def identity(x, beta=torch.tensor([0])):
    """Identity activation function.

    Definition:
    :math:`f(x) = x`
    """
    return x


@register_activation
# @torch.jit.script
def sigmoid_beta(x, beta=torch.tensor([10])):
    r"""Sigmoid activation function with scaling.

    Definition:
    :math:`f(x) = \frac{1}{1 + e^{-\beta x}}`

    Keyword Arguments:
        beta (`torch.tensor`): Scaling parameter :math:`\beta`
    """
    out = torch.reciprocal(1.0 + torch.exp(-int(beta.item()) * x))
    return out


@register_activation
# @torch.jit.script
def swish_beta(x, beta=torch.tensor([10])):
    r"""Swish activation function with scaling.

    Definition:
    :math:`f(x) = \frac{x}{1 + e^{-\beta x}}`

    Keyword Arguments:
        beta (`torch.tensor`): Scaling parameter :math:`\beta`
    """
    out = x * sigmoid_beta(x, beta=beta)
    return out


def get_activation(funcname):
    if callable(funcname):
        return funcname
    if funcname in ACTIVATIONS:
        return ACTIVATIONS.get(funcname)
    raise NameError(f'Activation {funcname} was not found.')
