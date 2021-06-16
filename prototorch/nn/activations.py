"""ProtoTorch activations"""

import torch

ACTIVATIONS = dict()


def register_activation(fn):
    """Add the activation function to the registry."""
    name = fn.__name__
    ACTIVATIONS[name] = fn
    return fn


@register_activation
def identity(x, beta=0.0):
    """Identity activation function.

    Definition:
    :math:`f(x) = x`

    Keyword Arguments:
        beta (`float`): Ignored.
    """
    return x


@register_activation
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
    elif funcname in ACTIVATIONS:
        return ACTIVATIONS.get(funcname)
    else:
        emsg = f"Unable to find matching function for `{funcname}` " \
            f"in `prototorch.nn.activations`. "
        helpmsg = f"Possible values are {list(ACTIVATIONS.keys())}."
        raise NameError(emsg + helpmsg)
