"""ProtoTorch abstract dataset classes.

Based on `torchvision.VisionDataset` and `torchvision.MNIST`

For the original code, see:
https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
"""

import os

import torch


class Dataset(torch.utils.data.Dataset):
    """Abstract dataset class to be inherited."""
    _repr_indent = 2

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ProtoDataset(Dataset):
    """Abstract dataset class to be inherited."""
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, download=True, verbose=True):
        super().__init__(root)
        self.train = train  # training set or test set
        self.verbose = verbose

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. '
                               'You can use download=True to download it')

        data_file = self.training_file if self.train else self.test_file

        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(
            os.path.join(self.processed_folder, self.training_file))
                and os.path.exists(
                    os.path.join(self.processed_folder, self.test_file)))

    def __repr__(self):
        head = 'Dataset ' + self.__class__.__name__
        body = ['Number of datapoints: {}'.format(self.__len__())]
        if self.root is not None:
            body.append('Root location: {}'.format(self.root))
        body += self.extra_repr().splitlines()
        lines = [head] + [' ' * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return f"Split: {'Train' if self.train is True else 'Test'}"

    def __len__(self):
        return len(self.data)

    def download(self):
        raise NotImplementedError
