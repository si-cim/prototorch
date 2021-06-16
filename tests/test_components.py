"""ProtoTorch components test suite."""

import torch

import prototorch as pt


def test_labcomps_zeros_init():
    protos = torch.zeros(3, 2)
    c = pt.components.LabeledComponents(
        distribution=[1, 1, 1],
        initializer=pt.components.Zeros(2),
    )
    assert (c.components == protos).any() == True


def test_labcomps_warmstart():
    protos = torch.randn(3, 2)
    plabels = torch.tensor([1, 2, 3])
    c = pt.components.LabeledComponents(
        distribution=[1, 1, 1],
        initializer=None,
        initialized_components=[protos, plabels],
    )
    assert (c.components == protos).any() == True
    assert (c.component_labels == plabels).any() == True
