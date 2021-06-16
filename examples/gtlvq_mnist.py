"""
ProtoTorch GTLVQ example using MNIST data.
The GTLVQ is placed as an classification model on
top of a CNN, considered as featurer extractor.
Initialization of subpsace and prototypes in
Siamnese fashion
For more info about GTLVQ see:
DOI:10.1109/IJCNN.2016.7727534
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from prototorch.functions.helper import calculate_prototype_accuracy
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.models import GTLVQ

# Parameters and options
num_epochs = 50
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.5
log_interval = 10
cuda = "cuda:0"
random_seed = 1
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

# Configures reproducability
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Prepare and preprocess the data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)


# Define the GLVQ model plus appropriate feature extractor
class CNNGTLVQ(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        subspace_data,
        prototype_data,
        tangent_projection_type="local",
        prototypes_per_class=2,
        bottleneck_dim=128,
    ):
        super(CNNGTLVQ, self).__init__()

        # Feature Extractor - Simple CNN
        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, bottleneck_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.LayerNorm(bottleneck_dim),
        )

        # Forward pass of subspace and prototype initialization data through feature extractor
        subspace_data = self.fe(subspace_data)
        prototype_data[0] = self.fe(prototype_data[0])

        # Initialization of GTLVQ
        self.gtlvq = GTLVQ(
            num_classes,
            subspace_data,
            prototype_data,
            tangent_projection_type=tangent_projection_type,
            feature_dim=bottleneck_dim,
            prototypes_per_class=prototypes_per_class,
        )

    def forward(self, x):
        # Feature Extraction
        x = self.fe(x)

        # GTLVQ Forward pass
        dis = self.gtlvq(x)
        return dis


# Get init data
subspace_data = torch.cat(
    [next(iter(train_loader))[0],
     next(iter(test_loader))[0]])
prototype_data = next(iter(train_loader))

# Build the CNN GTLVQ  model
model = CNNGTLVQ(
    10,
    subspace_data,
    prototype_data,
    tangent_projection_type="local",
    bottleneck_dim=128,
).to(device)

# Optimize using SGD optimizer from `torch.optim`
optimizer = torch.optim.Adam(
    [{
        "params": model.fe.parameters()
    }, {
        "params": model.gtlvq.parameters()
    }],
    lr=learning_rate,
)
criterion = GLVQLoss(squashing="sigmoid_beta", beta=10)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        model.train()
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()

        distances = model(x_train)
        plabels = model.gtlvq.cls.component_labels.to(device)

        # Compute loss.
        loss = criterion([distances, plabels], y_train)
        loss.backward()
        optimizer.step()

        # GTLVQ uses projected SGD, which means to orthogonalize the subspaces after every gradient update.
        model.gtlvq.orthogonalize_subspace()

        if batch_idx % log_interval == 0:
            acc = calculate_prototype_accuracy(distances, y_train, plabels)
            print(
                f"Epoch: {epoch + 1:02d}/{num_epochs:02d} Epoch Progress: {100. * batch_idx / len(train_loader):02.02f} % Loss: {loss.item():02.02f} \
              Train Acc: {acc.item():02.02f}")

    # Test
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            test_distances = model(torch.tensor(x_test))
            test_plabels = model.gtlvq.cls.prototype_labels.to(device)
            i = torch.argmin(test_distances, 1)
            correct += torch.sum(y_test == test_plabels[i])
            total += y_test.size(0)
        print("Accuracy of the network on the test images: %d %%" %
              (torch.true_divide(correct, total) * 100))

# Save the model
PATH = "./glvq_mnist_model.pth"
torch.save(model.state_dict(), PATH)
