# Copyright 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A set of useful tools.
"""

import torch
import torch.nn as nn
import math


def init_rnn(rnn, init_range, weights=None, biases=None):
    """ Orthogonal initialization of RNN. """
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    bound = 1 / math.sqrt(rnn.hidden_size)

    # init weights
    for w in weights:
        #nn.init.orthogonal(rnn._parameters[w])
        rnn._parameters[w].data.uniform_(-bound, bound)
        #rnn._parameters[w].data.uniform_(-init_range, init_range)
        #rnn._parameters[w].data.orthogonal_()
    # init biases
    for b in biases:
        p = rnn._parameters[b]
        n = p.size(0)
        p.data.fill_(0)
        # init bias for the reset gate in GRU
        p.data.narrow(0, 0, n // 3).fill_(0.0)


def init_rnn_cell(rnn, init_range):
    """ Orthogonal initialization of RNNCell. """
    init_rnn(rnn, init_range, ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_linear(linear, init_range):
    """ Uniform initialization of Linear. """
    linear.weight.data.uniform_(-init_range, init_range)
    linear.bias.data.fill_(0)


def init_cont(cont, init_range):
    """ Uniform initialization of a container. """
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


def make_mask(n, marked, value=-1000):
    """ Create a masked tensor. """
    mask = torch.Tensor(n).fill_(0)
    for i in marked:
        mask[i] = value
    return mask
