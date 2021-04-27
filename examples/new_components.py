#
# DATASET
#
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train, y_train = load_iris(return_X_y=True)
x_train = x_train[:, [0, 2]]
scaler.fit(x_train)
x_train = scaler.transform(x_train)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
num_classes = len(torch.unique(y_train))

#
# CREATE NEW COMPONENTS
#
from prototorch.components import *
from prototorch.components.initializers import *

unsupervised = Components(6, SelectionInitializer(x_train))
print(unsupervised())

prototypes = LabeledComponents(
    (3, 2), StratifiedSelectionInitializer(x_train, y_train))
print(prototypes())

components = ReasoningComponents(
    (3, 6), StratifiedSelectionInitializer(x_train, y_train))
print(components())

#
# TEST SERIALIZATION
#
import io

save = io.BytesIO()
torch.save(unsupervised, save)
save.seek(0)
serialized_unsupervised = torch.load(save)

assert torch.all(unsupervised.components == serialized_unsupervised.components
                 ), "Serialization of Components failed."

save = io.BytesIO()
torch.save(prototypes, save)
save.seek(0)
serialized_prototypes = torch.load(save)

assert torch.all(prototypes.components == serialized_prototypes.components
                 ), "Serialization of Components failed."
assert torch.all(prototypes.labels == serialized_prototypes.labels
                 ), "Serialization of Components failed."

save = io.BytesIO()
torch.save(components, save)
save.seek(0)
serialized_components = torch.load(save)

assert torch.all(components.components == serialized_components.components
                 ), "Serialization of Components failed."
assert torch.all(components.reasonings == serialized_components.reasonings
                 ), "Serialization of Components failed."
