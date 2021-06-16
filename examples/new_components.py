"""This example script shows the usage of the new components architecture.

Serialization/deserialization also works as expected.

"""

import torch

import prototorch as pt

ds = pt.datasets.Iris()

unsupervised = pt.components.Components(
    6,
    initializer=pt.initializers.ZCI(2),
)
print(unsupervised())

prototypes = pt.components.LabeledComponents(
    (3, 2),
    components_initializer=pt.initializers.SSCI(ds),
)
print(prototypes())

components = pt.components.ReasoningComponents(
    (3, 2),
    components_initializer=pt.initializers.SSCI(ds),
    reasonings_initializer=pt.initializers.PPRI(),
)
print(prototypes())

# Test Serialization
import io

save = io.BytesIO()
torch.save(unsupervised, save)
save.seek(0)
serialized_unsupervised = torch.load(save)

assert torch.all(unsupervised.components == serialized_unsupervised.components)

save = io.BytesIO()
torch.save(prototypes, save)
save.seek(0)
serialized_prototypes = torch.load(save)

assert torch.all(prototypes.components == serialized_prototypes.components)
assert torch.all(prototypes.labels == serialized_prototypes.labels)

save = io.BytesIO()
torch.save(components, save)
save.seek(0)
serialized_components = torch.load(save)

assert torch.all(components.components == serialized_components.components)
assert torch.all(components.reasonings == serialized_components.reasonings)
