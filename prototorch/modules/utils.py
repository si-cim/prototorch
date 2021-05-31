"""ProtoTorch utility modules."""

import torch


class LambdaLayer(torch.nn.Module):
    def __init__(self, fn, name=None):
        super().__init__()
        self.fn = fn
        self.name = name or fn.__name__ # lambda fns get <lambda>

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def extra_repr(self):
        return self.name
