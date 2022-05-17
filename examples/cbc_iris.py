"""ProtoTorch CBC example using 2D Iris data."""

import logging

import torch
from matplotlib import pyplot as plt

import prototorch as pt


class CBC(torch.nn.Module):

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.components_layer = pt.components.ReasoningComponents(
            distribution=[2, 1, 2],
            components_initializer=pt.initializers.SSCI(data, noise=0.1),
            reasonings_initializer=pt.initializers.PPRI(components_first=True),
        )

    def forward(self, x):
        components, reasonings = self.components_layer()
        sims = pt.similarities.euclidean_similarity(x, components)
        probs = pt.competitions.cbcc(sims, reasonings)
        return probs


class VisCBC2D():

    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train = pt.utils.parse_data_arg(data)
        self.title = "Components Visualization"
        self.fig = plt.figure(self.title)
        self.border = 0.1
        self.resolution = 100
        self.cmap = "viridis"

    def on_train_epoch_end(self):
        x_train, y_train = self.x_train, self.y_train
        _components = self.model.components_layer._components.detach()
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.axis("off")
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )
        ax.scatter(
            _components[:, 0],
            _components[:, 1],
            c="w",
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )
        x = torch.vstack((x_train, _components))
        mesh_input, xx, yy = pt.utils.mesh2d(x, self.border, self.resolution)
        with torch.no_grad():
            y_pred = self.model(
                torch.Tensor(mesh_input).type_as(_components)).argmax(1)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)
        plt.pause(0.2)


if __name__ == "__main__":
    train_ds = pt.datasets.Iris(dims=[0, 2])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    model = CBC(train_ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = pt.losses.MarginLoss(margin=0.1)
    vis = VisCBC2D(model, train_ds)

    for epoch in range(200):
        correct = 0.0
        for x, y in train_loader:
            y_oh = torch.eye(3)[y]
            y_pred = model(x)
            loss = criterion(y_pred, y_oh).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (y_pred.argmax(1) == y).float().sum(0)

        acc = 100 * correct / len(train_ds)
        logging.info(f"Epoch: {epoch} Accuracy: {acc:05.02f}%")
        vis.on_train_epoch_end()
