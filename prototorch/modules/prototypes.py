"""ProtoTorch prototype modules."""

import torch

from prototorch.functions.initializers import get_initializer


class AddPrototypes1D(torch.nn.Module):
    def __init__(self,
                 prototypes_per_class=1,
                 prototype_distribution=None,
                 prototype_initializer='ones',
                 data=None,
                 **kwargs):

        if data is None:
            if 'input_dim' not in kwargs:
                raise NameError('`input_dim` required if '
                                'no `data` is provided.')
            if prototype_distribution is not None:
                nclasses = sum(prototype_distribution)
            else:
                if 'nclasses' not in kwargs:
                    raise NameError('`prototype_distribution` required if '
                                    'both `data` and `nclasses` are not '
                                    'provided.')
                nclasses = kwargs.pop('nclasses')
            input_dim = kwargs.pop('input_dim')
            # input_shape = (input_dim, )
            x_train = torch.rand(nclasses, input_dim)
            y_train = torch.arange(nclasses)

        else:
            x_train, y_train = data
            x_train = torch.as_tensor(x_train)
            y_train = torch.as_tensor(y_train)

        super().__init__(**kwargs)
        self.prototypes_per_class = prototypes_per_class
        with torch.no_grad():
            if not prototype_distribution:
                num_classes = torch.unique(y_train).shape[0]
                self.prototype_distribution = torch.tensor(
                    [self.prototypes_per_class] * num_classes)
            else:
                self.prototype_distribution = torch.tensor(
                    prototype_distribution)
        self.prototype_initializer = get_initializer(prototype_initializer)
        prototypes, prototype_labels = self.prototype_initializer(
            x_train,
            y_train,
            prototype_distribution=self.prototype_distribution)
        self.prototypes = torch.nn.Parameter(prototypes)
        self.prototype_labels = prototype_labels

    def forward(self):
        return self.prototypes, self.prototype_labels
