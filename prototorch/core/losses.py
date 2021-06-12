"""ProtoTorch losses"""

import torch

from ..nn.activations import get_activation


# Helpers
def _get_matcher(targets, labels):
    """Returns a boolean tensor."""
    matcher = torch.eq(targets.unsqueeze(dim=1), labels)
    if labels.ndim == 2:
        # if the labels are one-hot vectors
        num_classes = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
    return matcher


def _get_dp_dm(distances, targets, plabels, with_indices=False):
    """Returns the d+ and d- values for a batch of distances."""
    matcher = _get_matcher(targets, plabels)
    not_matcher = torch.bitwise_not(matcher)

    inf = torch.full_like(distances, fill_value=float("inf"))
    d_matching = torch.where(matcher, distances, inf)
    d_unmatching = torch.where(not_matcher, distances, inf)
    dp = torch.min(d_matching, dim=-1, keepdim=True)
    dm = torch.min(d_unmatching, dim=-1, keepdim=True)
    if with_indices:
        return dp, dm
    return dp.values, dm.values


# GLVQ
def glvq_loss(distances, target_labels, prototype_labels):
    """GLVQ loss function with support for one-hot labels."""
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm)
    return mu


def lvq1_loss(distances, target_labels, prototype_labels):
    """LVQ1 loss function with support for one-hot labels.

    See Section 4 [Sado&Yamada]
    https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = dp
    mu[dp > dm] = -dm[dp > dm]
    return mu


def lvq21_loss(distances, target_labels, prototype_labels):
    """LVQ2.1 loss function with support for one-hot labels.

    See Section 4 [Sado&Yamada]
    https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = dp - dm

    return mu


# Probabilistic
def _get_class_probabilities(probabilities, targets, prototype_labels):
    # Create Label Mapping
    uniques = prototype_labels.unique(sorted=True).tolist()
    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}

    target_indices = torch.LongTensor(list(map(key_val.get, targets.tolist())))

    whole = probabilities.sum(dim=1)
    correct = probabilities[torch.arange(len(probabilities)), target_indices]
    wrong = whole - correct

    return whole, correct, wrong


def nllr_loss(probabilities, targets, prototype_labels):
    """Compute the Negative Log-Likelihood Ratio loss."""
    _, correct, wrong = _get_class_probabilities(probabilities, targets,
                                                 prototype_labels)

    likelihood = correct / wrong
    log_likelihood = torch.log(likelihood)
    return -1.0 * log_likelihood


def rslvq_loss(probabilities, targets, prototype_labels):
    """Compute the Robust Soft Learning Vector Quantization (RSLVQ) loss."""
    whole, correct, _ = _get_class_probabilities(probabilities, targets,
                                                 prototype_labels)

    likelihood = correct / whole
    log_likelihood = torch.log(likelihood)
    return -1.0 * log_likelihood


class GLVQLoss(torch.nn.Module):
    def __init__(self, margin=0.0, squashing="identity", beta=10, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.squashing = get_activation(squashing)
        self.beta = torch.tensor(beta)

    def forward(self, outputs, targets):
        distances, plabels = outputs
        mu = glvq_loss(distances, targets, prototype_labels=plabels)
        batch_loss = self.squashing(mu + self.margin, beta=self.beta)
        return torch.sum(batch_loss, dim=0)


class NeuralGasEnergy(torch.nn.Module):
    def __init__(self, lm, **kwargs):
        super().__init__(**kwargs)
        self.lm = lm

    def forward(self, d):
        order = torch.argsort(d, dim=1)
        ranks = torch.argsort(order, dim=1)
        cost = torch.sum(self._nghood_fn(ranks, self.lm) * d)

        return cost, order

    def extra_repr(self):
        return f"lambda: {self.lm}"

    @staticmethod
    def _nghood_fn(rankings, lm):
        return torch.exp(-rankings / lm)


class GrowingNeuralGasEnergy(NeuralGasEnergy):
    def __init__(self, topology_layer, **kwargs):
        super().__init__(**kwargs)
        self.topology_layer = topology_layer

    @staticmethod
    def _nghood_fn(rankings, topology):
        winner = rankings[:, 0]

        weights = torch.zeros_like(rankings, dtype=torch.float)
        weights[torch.arange(rankings.shape[0]), winner] = 1.0

        neighbours = topology.get_neighbours(winner)

        weights[neighbours] = 0.1

        return weights
