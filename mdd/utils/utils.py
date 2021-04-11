import numpy as np
import torch
import torch.nn as nn
from torch import linalg as la
from torch.autograd import Function


class GradientReverseLayer(Function):
    """
    Gradient reversal layer from http://arxiv.org/abs/1505.07818 used in DANN
    and MDD approaches.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def pair_norm(labels, features):
    norm = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                count += 1
                norm += la.norm(features[i] - features[j], ord=2, dim=0).sum()
    return norm / count


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """
    Compute coefficient for GRL
    """
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def init_weights(m):
    classname = m.__class__.__name__
    if (
        classname.find("Conv2d") != -1
        or classname.find("ConvTranspose2d") != -1
    ):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
