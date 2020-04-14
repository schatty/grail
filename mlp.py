from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)


def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        self.loss_f = nn.BCELoss()

        if num_class > 2:
            raise Exception("Multiclass classification is not supported yet!")

        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        logits = self.h2_weights(h1)
        y_ = F.sigmoid(logits)
        y_ = torch.mean(y_)

        var_w = torch.pow(torch.std(y_), 2) + 1e-6
        var_w = 1. / var_w

        if y is not None:
            y = y.float()

            loss = self.loss_f(y_, y)

            pred = (y_ >= 0.5).int()
            acc = float(pred == y.item())

            return logits, loss, acc, var_w
        else:
            return logits
