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
def identity(input, beta=torch.tensor([0])):
    """:math:`f(x) = x`"""
    return input


@register_activation
# @torch.jit.script
def sigmoid_beta(input, beta=torch.tensor([10])):
    """:math:`f(x) = \\frac{1}{1 + e^{-\\beta x}}`

    Keyword Arguments:
        beta (float): Parameter :math:`\\beta`
    """
    out = torch.reciprocal(1.0 + torch.exp(-int(beta.item()) * input))
    return out


@register_activation
# @torch.jit.script
def swish_beta(input, beta=torch.tensor([10])):
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
