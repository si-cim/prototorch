"""ProtoTorch loss functions."""

import torch


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


def glvq_loss(distances, target_labels, prototype_labels):
    """GLVQ loss function with support for one-hot labels."""
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm)
    return mu


def _get_dop_in_diopf(diff_to_thresh, matcher, not_matcher):
    zero = torch.tensor(0).type(torch.float)
    d_inner_pn = torch.where(diff_to_thresh.float() < zero, diff_to_thresh.float(), zero)
    d_inner_p = torch.where(matcher, d_inner_pn, zero)
    d_inner_n = torch.where(not_matcher, d_inner_pn, zero)
    d_outer_pn = torch.where(diff_to_thresh >= zero, diff_to_thresh, zero)
    d_outer_p = torch.where(matcher, d_outer_pn, zero)
    _zeros = torch.full(d_outer_p.shape, 0).type(torch.float)
    d_outer_p_free = torch.where(torch.min(d_inner_p, dim=1).values < zero, _zeros.T, d_outer_p.T).T
    d_op_in = torch.add(d_outer_p_free, d_inner_n)
    d_iopf = torch.add(d_outer_p_free, d_inner_p)
    return d_op_in, d_iopf
    
    
def one_class_classifier_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ OneClassClassifier loss function """
    zero = torch.tensor(0).type(torch.float)
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)
    # Optimizing False Positives and Negatives
    diff_to_thresh = torch.subtract(distances.T, theta_boundary).T
    d_op_in, d_iopf = _get_dop_in_diopf(diff_to_thresh, matcher, not_matcher)
    muf = d_op_in * torch.pow(-1., torch.LongTensor(not_matcher.type(torch.long)))
    # Optimizing Margin (theta_boundary)
    d_unmatching = torch.where(not_matcher, distances, zero)
    dp_max = torch.max(d_iopf, dim=-1, keepdim=True).values
    dn_min = torch.min(d_unmatching, dim=-1, keepdim=True).values
    mut = diff_to_thresh - torch.divide(torch.add(dp_max, dn_min),2)
    # Minimizing distances to True Positives (similar to penalty term)
    #d_matching_zero = torch.where(matcher, distances, 0)
    #mud = torch.min(d_matching_zero, dim=-1, keepdims=True)
    mud = torch.min(d_iopf, dim=-1, keepdims=True).values
    print(muf, mut, mud)
    return muf + mut + mud


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
