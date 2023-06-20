"""ProtoTorch GMLVQ example using Iris data."""

import torch

import prototorch as pt


class GMLVQ(torch.nn.Module):
    """
    Implementation of Generalized Matrix Learning Vector Quantization.
    """

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)

        self.components_layer = pt.components.LabeledComponents(
            distribution=[1, 1, 1],
            components_initializer=pt.initializers.SMCI(data, noise=0.1),
        )

        self.backbone = pt.transforms.Omega(
            len(data[0][0]),
            len(data[0][0]),
            pt.initializers.RandomLinearTransformInitializer(),
        )

    def forward(self, data):
        """
        Forward function that returns a tuple of dissimilarities and label information.
        Feed into GLVQLoss to get a complete GMLVQ model.
        """
        components, label = self.components_layer()

        latent_x = self.backbone(data)
        latent_components = self.backbone(components)

        distance = pt.distances.squared_euclidean_distance(
            latent_x, latent_components)

        return distance, label

    def predict(self, data):
        """
        The GMLVQ has a modified prediction step, where a competition layer is applied.
        """
        components, label = self.components_layer()
        distance = pt.distances.squared_euclidean_distance(data, components)
        winning_label = pt.competitions.wtac(distance, label)
        return winning_label


if __name__ == "__main__":
    train_ds = pt.datasets.Iris()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    model = GMLVQ(train_ds)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = pt.losses.GLVQLoss()

    for epoch in range(200):
        correct = 0.0
        for x, y in train_loader:
            d, labels = model(x)
            loss = criterion(d, y, labels).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred = model.predict(x)
                correct += (y_pred == y).float().sum(0)

        acc = 100 * correct / len(train_ds)
        print(f"Epoch: {epoch} Accuracy: {acc:05.02f}%")
