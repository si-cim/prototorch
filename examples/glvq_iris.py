"""ProtoTorch GLVQ example using 2D Iris data."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from prototorch.functions.competitions import wtac
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torchinfo import summary

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
        self.proto_layer = Prototypes1D(
            input_dim=2,
            prototypes_per_class=3,
            num_classes=3,
            prototype_initializer="stratified_random",
            data=[x_train, y_train],
        )

    def forward(self, x):
        protos = self.proto_layer.prototypes
        plabels = self.proto_layer.prototype_labels
        dis = euclidean_distance(x, protos)
        return dis, plabels


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
title = "Prototype Visualization"
fig = plt.figure(title)
for epoch in range(70):
    # Compute loss
    dis, plabels = model(x_in)
    loss = criterion([dis, plabels], y_in)
    with torch.no_grad():
        pred = wtac(dis, plabels)
        correct = pred.eq(y_in.view_as(pred)).sum().item()
    acc = 100.0 * correct / len(x_train)
    print(
        f"Epoch: {epoch + 1:03d} Loss: {loss.item():05.02f} Acc: {acc:05.02f}%"
    )

    # Take a gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get the prototypes form the model
    protos = model.proto_layer.prototypes.data.numpy()
    if np.isnan(np.sum(protos)):
        print("Stopping training because of `nan` in prototypes.")
        break

    # Visualize the data and the prototypes
    ax = fig.gca()
    ax.cla()
    ax.set_title(title)
    ax.set_xlabel("Data dimension 1")
    ax.set_ylabel("Data dimension 2")
    cmap = "viridis"
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor="k")
    ax.scatter(
        protos[:, 0],
        protos[:, 1],
        c=plabels,
        cmap=cmap,
        edgecolor="k",
        marker="D",
        s=50,
    )

    # Paint decision regions
    x = np.vstack((x_train, protos))
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / 50),
                         np.arange(y_min, y_max, 1 / 50))
    mesh_input = np.c_[xx.ravel(), yy.ravel()]

    torch_input = torch.Tensor(mesh_input)
    d = model(torch_input)[0]
    w_indices = torch.argmin(d, dim=1)
    y_pred = torch.index_select(plabels, 0, w_indices)
    y_pred = y_pred.reshape(xx.shape)

    # Plot voronoi regions
    ax.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.35)

    ax.set_xlim(left=x_min + 0, right=x_max - 0)
    ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

    plt.pause(0.1)
