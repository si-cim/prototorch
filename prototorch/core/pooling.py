"""ProtoTorch pooling"""

from typing import Callable

import torch


def stratify_with(values: torch.Tensor,
                  labels: torch.LongTensor,
                  fn: Callable,
                  fill_value: float = 0.0) -> (torch.Tensor):
    """Apply an arbitrary stratification strategy on the columns on `values`.

    The outputs correspond to sorted labels.
    """
    clabels = torch.unique(labels, dim=0, sorted=True)
    num_classes = clabels.size()[0]
    if values.size()[1] == num_classes:
        # skip if stratification is trivial
        return values
    batch_size = values.size()[0]
    winning_values = torch.zeros(num_classes, batch_size, device=labels.device)
    filler = torch.full_like(values.T, fill_value=fill_value)
    for i, cl in enumerate(clabels):
        matcher = torch.eq(labels.unsqueeze(dim=1), cl)
        if labels.ndim == 2:
            # if the labels are one-hot vectors
            matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
        cdists = torch.where(matcher, values.T, filler).T
        winning_values[i] = fn(cdists)
    if labels.ndim == 2:
        # Transpose to return with `batch_size` first and
        # reverse the columns to fix the ordering of the classes
        return torch.flip(winning_values.T, dims=(1, ))

    return winning_values.T  # return with `batch_size` first


def stratified_sum_pooling(values: torch.Tensor,
                           labels: torch.LongTensor) -> (torch.Tensor):
    """Group-wise sum."""
    winning_values = stratify_with(
        values,
        labels,
        fn=lambda x: torch.sum(x, dim=1, keepdim=True).squeeze(),
        fill_value=0.0)
    return winning_values


def stratified_min_pooling(values: torch.Tensor,
                           labels: torch.LongTensor) -> (torch.Tensor):
    """Group-wise minimum."""
    winning_values = stratify_with(
        values,
        labels,
        fn=lambda x: torch.min(x, dim=1, keepdim=True).values.squeeze(),
        fill_value=float("inf"))
    return winning_values


def stratified_max_pooling(values: torch.Tensor,
                           labels: torch.LongTensor) -> (torch.Tensor):
    """Group-wise maximum."""
    winning_values = stratify_with(
        values,
        labels,
        fn=lambda x: torch.max(x, dim=1, keepdim=True).values.squeeze(),
        fill_value=-1.0 * float("inf"))
    return winning_values


def stratified_prod_pooling(values: torch.Tensor,
                            labels: torch.LongTensor) -> (torch.Tensor):
    """Group-wise maximum."""
    winning_values = stratify_with(
        values,
        labels,
        fn=lambda x: torch.prod(x, dim=1, keepdim=True).squeeze(),
        fill_value=1.0)
    return winning_values


class StratifiedSumPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_sum_pooling` function."""
    def forward(self, values, labels):  # pylint: disable=no-self-use
        return stratified_sum_pooling(values, labels)


class StratifiedProdPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_prod_pooling` function."""
    def forward(self, values, labels):  # pylint: disable=no-self-use
        return stratified_prod_pooling(values, labels)


class StratifiedMinPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_min_pooling` function."""
    def forward(self, values, labels):  # pylint: disable=no-self-use
        return stratified_min_pooling(values, labels)


class StratifiedMaxPooling(torch.nn.Module):
    """Thin wrapper over the `stratified_max_pooling` function."""
    def forward(self, values, labels):  # pylint: disable=no-self-use
        return stratified_max_pooling(values, labels)
