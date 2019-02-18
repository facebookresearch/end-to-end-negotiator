# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

import data
from engines.selection_engine import SelectionEngine
from domain import get_domain
from models.utils import *
from models.ctx_encoder import MlpContextEncoder
from models.attn import Attention, HierarchicalAttention


class SelectionModule(nn.Module):
    def __init__(self, query_size, value_size, hidden_size, selection_size, num_heads, output_size, args):
        super(SelectionModule, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = HierarchicalAttention(query_size, value_size, hidden_size,
            args.dropout, args.init_range)

        self.sel_encoder = nn.Sequential(
            nn.Linear(2 * hidden_size + query_size, selection_size),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(args.dropout)

        self.sel_decoders = nn.ModuleList()
        for i in range(num_heads):
            self.sel_decoders.append(nn.Linear(selection_size, output_size))

        init_cont(self.sel_encoder, args.init_range)
        init_cont(self.sel_decoders, args.init_range)

    def flatten_parameters(self):
        self.attn.flatten_parameters()

    def forward(self, q, hs, lens, rev_idxs, hid_idxs):
        # run attention over hs, condition on q
        (h, sent_p), (_, word_ps) = self.attn(q, hs, lens, rev_idxs, hid_idxs)

        # add q to h
        h = torch.cat([h, q], 1)

        h = self.sel_encoder(h)
        h = self.dropout(h)

        outs = [decoder(h) for decoder in self.sel_decoders]
        outs = torch.cat(outs, 1).view(-1, self.output_size)
        return outs, sent_p, word_ps


class SelectionModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = SelectionEngine
    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(SelectionModel, self).__init__()

        self.nhid_pos = 32
        self.nhid_speaker = 32
        self.len_cutoff = 10

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.word_encoder = nn.Embedding(len(self.word_dict), args.nembed_word)
        self.pos_encoder = nn.Embedding(self.len_cutoff, self.nhid_pos)
        self.speaker_encoder = nn.Embedding(len(self.word_dict), self.nhid_speaker)
        self.ctx_encoder = MlpContextEncoder(len(self.context_dict), domain.input_length(),
            args.nembed_ctx, args.nhid_ctx, args.dropout, args.init_range, args.skip_values)

        self.sel_head = SelectionModule(
            query_size=args.nhid_ctx,
            value_size=args.nembed_word + self.nhid_pos + self.nhid_speaker,
            hidden_size=args.nhid_attn,
            selection_size=args.nhid_sel,
            num_heads=6,
            output_size=len(item_dict),
            args=args)

        self.dropout = nn.Dropout(args.dropout)

        # init embeddings
        self.word_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)
        self.pos_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)
        self.speaker_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)

    def flatten_parameters(self):
        self.sel_head.flatten_parameters()

    def forward_inpts(self, inpts, ctx_h):
        hs = []
        for i, inpt in enumerate(inpts):
            speaker_emb = self.speaker_encoder(inpt[0]).unsqueeze(0)
            inpt_emb = self.word_encoder(inpt)
            inpt_emb = self.dropout(inpt_emb)
            pos = Variable(torch.Tensor([min(self.len_cutoff, len(inpts) - i) - 1]).long())
            pos_emb = self.pos_encoder(pos).unsqueeze(0)

            # duplicate ctx_h along the temporal dimension and cat it with the input
            h = torch.cat([
                inpt_emb,
                speaker_emb.expand(inpt_emb.size(0), speaker_emb.size(1), speaker_emb.size(2)),
                pos_emb.expand(inpt_emb.size(0), inpt_emb.size(1), pos_emb.size(2))],
                2)
            hs.append(h)

        return hs

    def forward(self, inpts, lens, rev_idxs, hid_idxs, ctx):
        ctx_h = self.ctx_encoder(ctx)
        ctx_h = self.dropout(ctx_h)
        hs = self.forward_inpts(inpts, ctx_h)
        sel, _, _ = self.sel_head(ctx_h, hs, lens, rev_idxs, hid_idxs)
        return sel

    def forward_each_timestamp(self, inpts, lens, rev_idxs, hid_idxs, ctx):
        ctx_h = self.ctx_encoder(ctx)
        ctx_h = self.dropout(ctx_h)
        hs = self.forward_inpts(inpts, ctx_h)
        sels = []
        for i in range(len(hs)):
            sel, _, _ = self.sel_head(ctx_h, hs[:i + 1], lens[: i + 1],
                rev_idxs[:i + 1], hid_idxs[:i + 1])
            sels.append(sel)
        return sels





