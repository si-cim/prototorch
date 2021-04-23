"""ProtoTorch "siamese" GMLVQ example using Tecator."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from prototorch.datasets.tecator import Tecator
from prototorch.functions.distances import sed
from prototorch.modules import Prototypes1D
from prototorch.modules.losses import GLVQLoss
from prototorch.utils.colors import get_legend_handles

# Prepare the dataset and dataloader
train_data = Tecator(root="./artifacts", train=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)


class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        """GMLVQ model as a siamese network."""
        super().__init__()
        x, y = train_data.data, train_data.targets
        self.p1 = Prototypes1D(
            input_dim=100,
            prototypes_per_class=2,
            nclasses=2,
            prototype_initializer="stratified_random",
            data=[x, y],
        )
        self.omega = torch.nn.Linear(in_features=100,
                                     out_features=100,
                                     bias=False)
        torch.nn.init.eye_(self.omega.weight)

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels

        # Process `x` and `protos` through `omega`
        x_map = self.omega(x)
        protos_map = self.omega(protos)

        # Compute distances and output
        dis = sed(x_map, protos_map)
        return dis, plabels


# Build the GLVQ model
model = Model()

# Print a summary of the model
print(model)

# Optimize using Adam optimizer from `torch.optim`
optimizer = torch.optim.Adam(model.parameters(), lr=0.001_0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
criterion = GLVQLoss(squashing="identity", beta=10)

# Training loop
for epoch in range(150):
    epoch_loss = 0.0  # zero-out epoch loss
    optimizer.zero_grad()  # zero-out gradients
    for xb, yb in train_loader:
        # Compute loss
        distances, plabels = model(xb)
        loss = criterion([distances, plabels], yb)
        epoch_loss += loss.item()
        # Backprop
        loss.backward()
    # Take a gradient descent step
    optimizer.step()
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch + 1:03d} Loss: {epoch_loss:06.02f} lr: {lr:07.06f}")

# Get the omega matrix form the model
omega = model.omega.weight.data.numpy().T

# Visualize the lambda matrix
title = "Lambda Matrix Visualization"
fig = plt.figure(title)
ax = fig.gca()
ax.set_title(title)
im = ax.imshow(omega.dot(omega.T), cmap="viridis")
plt.show()

# Get the prototypes form the model
protos = model.p1.prototypes.data.numpy()
plabels = model.p1.prototype_labels

# Visualize the prototypes
title = "Tecator Prototypes"
fig = plt.figure(title)
ax = fig.gca()
ax.set_title(title)
ax.set_xlabel("Spectral frequencies")
ax.set_ylabel("Absorption")
clabels = ["Class 0 - Low fat", "Class 1 - High fat"]
handles, colors = get_legend_handles(clabels, marker="line", zero_indexed=True)
for x, y in zip(protos, plabels):
    ax.plot(x, c=colors[int(y)])
ax.legend(handles, clabels)
plt.show()
