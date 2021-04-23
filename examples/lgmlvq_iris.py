"""ProtoTorch LGMLVQ example using 2D Iris data."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from prototorch.functions.competitions import stratified_min
from prototorch.functions.distances import lomega_distance
from prototorch.functions.init import eye_
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D

# Prepare training data
x_train, y_train = load_iris(True)
x_train = x_train[:, [0, 2]]


# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        """Local-GMLVQ model."""
        super().__init__()
        self.p1 = Prototypes1D(
            input_dim=2,
            prototype_distribution=[1, 2, 2],
            prototype_initializer="stratified_random",
            data=[x_train, y_train],
        )
        omegas = torch.zeros(5, 2, 2)
        self.omegas = torch.nn.Parameter(omegas)
        eye_(self.omegas)

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        omegas = self.omegas
        dis = lomega_distance(x, protos, omegas)
        return dis, plabels


# Build the model
model = Model()

# Optimize using Adam optimizer from `torch.optim`
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = GLVQLoss(squashing="sigmoid_beta", beta=10)

x_in = torch.Tensor(x_train)
y_in = torch.Tensor(y_train)

# Training loop
title = "Prototype Visualization"
fig = plt.figure(title)
for epoch in range(100):
    # Compute loss
    dis, plabels = model(x_in)
    loss = criterion([dis, plabels], y_in)
    y_pred = np.argmin(stratified_min(dis, plabels).detach().numpy(), axis=1)
    acc = accuracy_score(y_train, y_pred)
    log_string = f"Epoch: {epoch + 1:03d} Loss: {loss.item():05.02f} "
    log_string += f"Acc: {acc * 100:05.02f}%"
    print(log_string)

    # Take a gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get the prototypes form the model
    protos = model.p1.prototypes.data.numpy()

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

    d, plabels = model(torch.Tensor(mesh_input))
    y_pred = np.argmin(stratified_min(d, plabels).detach().numpy(), axis=1)
    y_pred = y_pred.reshape(xx.shape)

    # Plot voronoi regions
    ax.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.35)

    ax.set_xlim(left=x_min + 0, right=x_max - 0)
    ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

    plt.pause(0.1)
