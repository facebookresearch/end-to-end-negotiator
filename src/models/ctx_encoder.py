# Copyright 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Set of context encoders.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from models.utils import *


class MlpContextEncoder(nn.Module):
    """ Simple encoder for the dialogue context. Encoder counts and values via MLP. """
    def __init__(self, n, k, nembed, nhid, dropout, init_range, skip_values=False):
        super(MlpContextEncoder, self).__init__()

        # embeddings for counts and values
        self.cnt_enc = nn.Sequential(
            nn.Embedding(n, nembed),
            nn.Dropout(dropout))
        self.val_enc = nn.Sequential(
            nn.Embedding(n, nembed),
            nn.Dropout(dropout))

        self.encoder = nn.Sequential(
            nn.Linear(k * nembed, nhid),
            nn.Tanh()
        )

        # a flag to only use counts to encode the context
        self.skip_values = skip_values

        init_cont(self.cnt_enc, init_range)
        init_cont(self.val_enc, init_range)
        init_cont(self.encoder, init_range)

    def forward(self, ctx):
        cnt_idx = Variable(torch.Tensor(range(0, ctx.size(0), 2)).long())
        cnt = ctx.index_select(0, cnt_idx)
        cnt_emb = self.cnt_enc(cnt)

        if self.skip_values:
            h = cnt_emb
        else:
            val_idx = Variable(torch.Tensor(range(1, ctx.size(0), 2)).long())
            val = ctx.index_select(0, val_idx)
            val_emb = self.val_enc(val)
            # element vise multiplication of embeddings
            h = torch.mul(cnt_emb, val_emb)

        # run MLP to acquire fixed size representation
        h = h.transpose(0, 1).contiguous().view(ctx.size(1), -1)
        ctx_h = self.encoder(h)
        return ctx_h
