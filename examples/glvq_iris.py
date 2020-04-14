"""ProtoTorch GLVQ example using 2D Iris data."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D

# Prepare and preprocess the data
scaler = StandardScaler()
x_train, y_train = load_iris(True)
x_train = x_train[:, [0, 2]]
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# Define the GLVQ model
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        """GLVQ model."""
        super().__init__()
        self.p1 = Prototypes1D(input_dim=2,
                               prototypes_per_class=1,
                               nclasses=3,
                               prototype_initializer='zeros')

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        dis = euclidean_distance(x, protos)
        return dis, plabels


# Build the GLVQ model
model = Model()

# Optimize using SGD optimizer from `torch.optim`
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = GLVQLoss(squashing='sigmoid_beta', beta=10)

x_in = torch.Tensor(x_train)
y_in = torch.Tensor(y_train)

# Training loop
title = 'Prototype Visualization'
fig = plt.figure(title)
for epoch in range(70):
    # Compute loss
    dis, plabels = model(x_in)
    loss = criterion([dis, plabels], y_in)
    print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():05.02f}')

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
    ax.set_xlabel('Data dimension 1')
    ax.set_ylabel('Data dimension 2')
    cmap = 'viridis'
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolor='k')
    ax.scatter(protos[:, 0],
               protos[:, 1],
               c=plabels,
               cmap=cmap,
               edgecolor='k',
               marker='D',
               s=50)

    # Paint decision regions
    x = np.vstack((x_train, protos))
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / 50),
                         np.arange(y_min, y_max, 1 / 50))
    mesh_input = np.c_[xx.ravel(), yy.ravel()]

    torch_input = torch.Tensor(mesh_input)
    d = model(torch_input)[0]
    y_pred = np.argmin(d.detach().numpy(),
                       axis=1)  # assume one prototype per class
    y_pred = y_pred.reshape(xx.shape)

    # Plot voronoi regions
    ax.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.35)

    ax.set_xlim(left=x_min + 0, right=x_max - 0)
    ax.set_ylim(bottom=y_min + 0, top=y_max - 0)

    plt.pause(0.1)
