"""ProtoTorch GLVQ example using 2D Iris data."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torchinfo import summary

from prototorch.components import LabeledComponents, StratifiedMeanInitializer
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss

# Prepare and preprocess the data
scaler = StandardScaler()
x_train, y_train = load_iris(return_X_y=True)
x_train = x_train[:, [0, 2]]
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# Define the GLVQ model
class Model(torch.nn.Module):
    def __init__(self):
        """GLVQ model for training on 2D Iris data."""
        super().__init__()
        prototype_initializer = StratifiedMeanInitializer([x_train, y_train])
        prototype_distribution = {"num_classes": 3, "prototypes_per_class": 3}
        self.proto_layer = LabeledComponents(
            prototype_distribution,
            prototype_initializer,
        )

    def forward(self, x):
        prototypes, prototype_labels = self.proto_layer()
        distances = euclidean_distance(x, prototypes)
        return distances, prototype_labels


# Build the GLVQ model
model = Model()

# Print summary using torchinfo (might be buggy/incorrect)
print(summary(model))

# Optimize using SGD optimizer from `torch.optim`
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = GLVQLoss(squashing="sigmoid_beta", beta=10)

x_in = torch.Tensor(x_train)
y_in = torch.Tensor(y_train)

# Training loop
TITLE = "Prototype Visualization"
fig = plt.figure(TITLE)
for epoch in range(70):
    # Compute loss
    distances, prototype_labels = model(x_in)
    loss = criterion([distances, prototype_labels], y_in)

    # Compute Accuracy
    with torch.no_grad():
        predictions = wtac(distances, prototype_labels)
        correct = predictions.eq(y_in.view_as(predictions)).sum().item()
    acc = 100.0 * correct / len(x_train)

    print(
        f"Epoch: {epoch + 1:03d} Loss: {loss.item():05.02f} Acc: {acc:05.02f}%"
    )

    # Optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get the prototypes form the model
    prototypes = model.proto_layer.components.numpy()
    if np.isnan(np.sum(prototypes)):
        print("Stopping training because of `nan` in prototypes.")
        break

    # Visualize the data and the prototypes
    ax = fig.gca()
    ax.cla()
    ax.set_title(TITLE)
    ax.set_xlabel("Data dimension 1")
    ax.set_ylabel("Data dimension 2")
    cmap = "viridis"
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
    ax.scatter(
        prototypes[:, 0],
        prototypes[:, 1],
        c=prototype_labels,
        cmap=cmap,
        edgecolor="k",
        marker="D",
        s=50,
    )

    # Paint decision regions
    x = np.vstack((x_train, prototypes))
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / 50),
                         np.arange(y_min, y_max, 1 / 50))
    mesh_input = np.c_[xx.ravel(), yy.ravel()]

    torch_input = torch.Tensor(mesh_input)
    d = model(torch_input)[0]
    w_indices = torch.argmin(d, dim=1)
    y_pred = torch.index_select(prototype_labels, 0, w_indices)
    y_pred = y_pred.reshape(xx.shape)

    # Plot voronoi regions
    ax.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.35)

    ax.set_xlim(left=x_min + 0, right=x_max - 0)
    ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

    plt.pause(0.1)
