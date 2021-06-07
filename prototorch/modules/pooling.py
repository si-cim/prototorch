"""ProtoTorch Pooling Modules."""

import torch
from prototorch.functions.pooling import (stratified_max_pooling,
                                          stratified_min_pooling,
                                          stratified_prod_pooling,
                                          stratified_sum_pooling)


class StratifiedSumPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_sum_pooling` function."""
    def forward(self, values, labels):
        return stratified_sum_pooling(values, labels)


class StratifiedProdPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_prod_pooling` function."""
    def forward(self, values, labels):
        return stratified_prod_pooling(values, labels)


class StratifiedMinPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_min_pooling` function."""
    def forward(self, values, labels):
        return stratified_min_pooling(values, labels)


class StratifiedMaxPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_max_pooling` function."""
    def forward(self, values, labels):
        return stratified_max_pooling(values, labels)
