import torch
from torch import linalg as la
from torch.nn import functional as F

from .utils.utils import pair_norm


def mdd_loss(features, labels, left_weight=1, right_weight=1):
    softmax_out = F.softmax(features, dim=1)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception("Incorrect batch size provided")

    batch_left = softmax_out[: int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size) :]

    loss = la.norm(batch_left - batch_right, ord=2, dim=1).sum() / float(
        batch_size
    )

    labels_left = labels[: int(0.5 * batch_size)]
    batch_left_loss = pair_norm(labels_left, batch_left)

    labels_right = labels[int(0.5 * batch_size) :]
    batch_right_loss = pair_norm(labels_right, batch_right)
    return (
        loss + left_weight * batch_left_loss + right_weight * batch_right_loss
    )


def entropic_loss(features):
    softmax_out = F.softmax(features, dim=1)
    batch_size = features.size(0)
    entropic_loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (
        1.0 / batch_size
    )
    return entropic_loss
