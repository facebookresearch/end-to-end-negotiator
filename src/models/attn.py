# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

from models.utils import *


class Attention(nn.Module):
    def __init__(self, query_size, value_size, hid_size, init_range):
        super(Attention, self).__init__()

        self.value2hid = nn.Linear(value_size, hid_size)
        self.query2hid = nn.Linear(query_size, hid_size)
        self.hid2output = nn.Linear(hid_size, 1)

        self.value2hid.weight.data.uniform_(-init_range, init_range)
        self.value2hid.bias.data.fill_(0)
        self.query2hid.weight.data.uniform_(-init_range, init_range)
        self.query2hid.bias.data.fill_(0)
        self.hid2output.weight.data.uniform_(-init_range, init_range)
        self.hid2output.bias.data.fill_(0)

    def _bottle(self, linear, x):
        y = linear(x.view(-1, x.size(-1)))
        return y.view(x.size(0), x.size(1), -1)

    def forward_attn(self, h):
        logit = self.attn(h.view(-1, h.size(2))).view(h.size(0), h.size(1))
        return logit

    def forward(self, q, v, mask=None):
        # q: [batch_size, query_size]
        # v: [length, batch_size, value_size]
        v = v.transpose(0, 1).contiguous()

        h_v = self._bottle(self.value2hid, v)
        h_q = self.query2hid(q)

        h = torch.tanh(h_v + h_q.unsqueeze(1).expand_as(h_v))
        logit = self._bottle(self.hid2output, h).squeeze(2)

        logit = logit.sub(logit.max(1, keepdim=True)[0].expand_as(logit))
        if mask is not None:
            logit = torch.add(logit, Variable(mask))

        p = F.softmax(logit, dim=1)
        w = p.unsqueeze(2).expand_as(v)
        h = torch.sum(torch.mul(v, w), 1, keepdim=True)
        h = h.transpose(0, 1).contiguous()

        return h, p


class KeyValueAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, hid_size, init_range):
        super(KeyValueAttention, self).__init__()

        self.key2hid = nn.Linear(key_size, hid_size)
        self.query2hid = nn.Linear(query_size, hid_size)
        self.hid2output = nn.Linear(hid_size, 1)

        self.key2hid.weight.data.uniform_(-init_range, init_range)
        self.key2hid.bias.data.fill_(0)
        self.query2hid.weight.data.uniform_(-init_range, init_range)
        self.query2hid.bias.data.fill_(0)
        self.hid2output.weight.data.uniform_(-init_range, init_range)
        self.hid2output.bias.data.fill_(0)

    def _bottle(self, linear, x):
        y = linear(x.view(-1, x.size(-1)))
        return y.view(x.size(0), x.size(1), -1)

    def forward_attn(self, h):
        logit = self.attn(h.view(-1, h.size(2))).view(h.size(0), h.size(1))
        return logit

    def forward(self, q, k, v, mask=None):
        # q: [batch_size, query_size]
        # k: [length, batch_size, key_size]
        # v: [length, batch_size, value_size]
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        h_k = self._bottle(self.key2hid, k)
        h_q = self.query2hid(q)

        h = F.tanh(h_k + h_q.unsqueeze(1).expand_as(h_k))
        logit = self._bottle(self.hid2output, h).squeeze(2)

        logit = logit.sub(logit.max(1, keepdim=True)[0].expand_as(logit))
        if mask is not None:
            logit = torch.add(logit, Variable(mask))

        p = F.softmax(logit)
        w = p.unsqueeze(2).expand_as(v)
        h = torch.sum(torch.mul(v, w), 1, keepdim=True)
        h = h.transpose(0, 1).contiguous()

        return h, p


class MaskedAttention(nn.Module):
    def __init__(self, query_size, value_size, hid_size, init_range):
        super(MaskedAttention, self).__init__()
        self.attn = Attention(query_size, value_size, hid_size, init_range)

    def make_mask(self, v, ln):
        mask = torch.Tensor(v.size(1), v.size(0)).fill_(-1000)
        for i in range(v.size(1)):
            mask.narrow(0, i, 1).narrow(1, 0, ln[i] + 1).fill_(0)
        return mask

    def forward(self, q, v, ln=None):
        mask = self.make_mask(v, ln) if ln is not None else None
        return self.attn(q, v, mask=mask)


class ChunkedAttention(nn.Module):
    def __init__(self, query_size, value_size, hid_size, init_range):
        super(ChunkedAttention, self).__init__()
        self.query_size = query_size
        self.value_size = value_size
        self.hid_size = hid_size
        self.init_range = init_range

    def zero_h(self, bsz, n=1):
        h = torch.Tensor(n, bsz, self.hid_size).fill_(0)
        return Variable(h)

    def make_mask(self, fwd_hs, lens):
        bsz = fwd_hs[0].size(1)
        n = sum(fwd_h.size(0) for fwd_h in fwd_hs)

        mask = torch.Tensor(bsz, n).fill_(-1000)
        offset = 0
        for fwd_h, ln in zip(fwd_hs, lens):
            for i in range(bsz):
                mask.narrow(0, i, 1).narrow(1, offset, ln[i] + 1).fill_(0)
            offset += fwd_h.size(0)
        return mask

    def reverse(self, fwd_inpts, fwd_lens, rev_idxs):
        bwd_inpts, bwd_lens = [], []
        for inpt, ln, rev_idx in zip(reversed(fwd_inpts), reversed(fwd_lens), reversed(rev_idxs)):
            bwd_inpt = inpt.gather(0,
                rev_idx.expand(rev_idx.size(0), rev_idx.size(1), inpt.size(2)))
            bwd_inpts.append(bwd_inpt)
            bwd_lens.append(ln)

        return bwd_inpts, bwd_lens


class BiRnnAttention(ChunkedAttention):
    def __init__(self, query_size, value_size, hid_size, dropout, init_range):
        super(BiRnnAttention, self).__init__(query_size, value_size, hid_size, init_range)

        self.dropout = nn.Dropout(dropout)

        self.fwd_rnn = nn.GRU(value_size, hid_size, bias=True)
        self.bwd_rnn = nn.GRU(value_size, hid_size, bias=True)

        self.attn = Attention(query_size, 2 * hid_size, hid_size, init_range)

    def flatten_parameters(self):
        self.fwd_rnn.flatten_parameters()
        self.bwd_rnn.flatten_parameters()

    def forward_attn(self, query, fwd_hs, bwd_hs, lens):
        fwd_h = torch.cat(fwd_hs, 0)
        bwd_h = torch.cat(bwd_hs, 0)
        h = torch.cat([fwd_h, bwd_h], 2)
        h = self.dropout(h)

        mask = self.make_mask(fwd_hs, lens)
        h, p = self.attn(query, h, mask)
        return h, p

    def forward_rnn(self, rnn, inpts, lens, hid_idxs):
        bsz = inpts[0].size(1)
        hs = []
        h = self.zero_h(bsz)
        for inpt, ln, hid_idx in zip(inpts, lens, hid_idxs):
            out, _ = rnn(inpt, h)
            hs.append(out)
            h = out.gather(0, hid_idx.expand(hid_idx.size(0), hid_idx.size(1), out.size(2)))
        return hs

    def forward(self, query, fwd_inpts, fwd_lens, rev_idxs, hid_idxs):
        # reverse inputs
        bwd_inpts, bwd_lens = self.reverse(fwd_inpts, fwd_lens, rev_idxs)

        fwd_hs = self.forward_rnn(self.fwd_rnn, fwd_inpts, fwd_lens, hid_idxs)
        bwd_hs = self.forward_rnn(self.bwd_rnn, bwd_inpts, bwd_lens, reversed(hid_idxs))

        # reverse them back to align with fwd_inpts
        bwd_hs, _ = self.reverse(bwd_hs, bwd_lens, list(reversed(rev_idxs)))

        h, p = self.forward_attn(query, fwd_hs, bwd_hs, fwd_lens)
        h = h.squeeze(0)
        return h, p


class HierarchicalAttention(ChunkedAttention):
    def __init__(self, query_size, value_size, hid_size, dropout, init_range):
        super(HierarchicalAttention, self).__init__(
            query_size, value_size, hid_size, init_range)

        self.word_dropout = nn.Dropout(dropout)
        self.sent_dropout = nn.Dropout(dropout)

        self.fwd_word_rnn = nn.GRU(value_size, hid_size, bias=True)
        self.bwd_word_rnn = nn.GRU(value_size, hid_size, bias=True)
        self.word_attn = Attention(query_size, 2 * hid_size, hid_size, init_range)

        self.sent_rnn = nn.GRU(2 * hid_size, hid_size, bidirectional=True, bias=True)
        self.sent_attn = Attention(query_size, 2 * hid_size, hid_size, init_range)

        init_rnn(self.fwd_word_rnn, init_range)
        init_rnn(self.bwd_word_rnn, init_range)
        init_rnn(self.sent_rnn, init_range)

    def flatten_parameters(self):
        self.fwd_word_rnn.flatten_parameters()
        self.bwd_word_rnn.flatten_parameters()
        self.sent_rnn.flatten_parameters()

    def forward_word_attn(self, query, fwd_word_hs, bwd_word_hs, ln, rev_idx, hid_idx):
        # reverse bwd_word_h
        bwd_word_hs = bwd_word_hs.gather(0,
            rev_idx.expand(rev_idx.size(0), rev_idx.size(1), bwd_word_hs.size(2)))

        word_hs = torch.cat([fwd_word_hs, bwd_word_hs], 2)
        word_hs = self.word_dropout(word_hs)

        mask = self.make_mask([fwd_word_hs], [ln])
        word_h, word_p = self.word_attn(query, word_hs, mask)
        return word_h, word_p

    def forward_word_rnn(self, rnn, bsz, inpts, lens, rev_idxs, hid_idxs):
        hs = []
        for inpt, ln, rev_idx, hid_idx in zip(inpts, lens, rev_idxs, hid_idxs):
            h, _ = rnn(inpt, self.zero_h(bsz))
            hs.append(h)
        return hs

    def forward_sent_attn(self, query, word_hs):
        bsz = query.size(0)
        if len(word_hs) == 0:
            sent_h = Variable(torch.Tensor(1, bsz, 2 * self.hid_size).fill_(0))
        else:
            word_hs = torch.cat(word_hs, 0)

            sent_h, _ = self.sent_rnn(word_hs, self.zero_h(bsz, n=2))
            sent_h = self.sent_dropout(sent_h)

        sent_h, sent_p = self.sent_attn(query, sent_h)
        return sent_h, sent_p

    def forward(self, query, fwd_inpts, fwd_lens, rev_idxs, hid_idxs):
        # query: [batch_size, query_size]
        # fwd_inpts: [length, batch_size, value_size]
        bwd_inpts, bwd_lens = self.reverse(fwd_inpts, fwd_lens, rev_idxs)

        bsz = query.size(0)
        fwd_word_hs = self.forward_word_rnn(self.fwd_word_rnn, bsz, fwd_inpts,
            fwd_lens, rev_idxs, hid_idxs)
        bwd_word_hs = self.forward_word_rnn(self.bwd_word_rnn, bsz, bwd_inpts,
            reversed(fwd_lens), reversed(rev_idxs), reversed(hid_idxs))

        iterator = zip(fwd_word_hs, reversed(bwd_word_hs), fwd_lens, rev_idxs, hid_idxs)
        word_hs, word_ps = [], []
        for fwd_word_h, bwd_word_h, ln, rev_idx, hid_idx in iterator:
            word_h, word_p = self.forward_word_attn(query, fwd_word_h,
                bwd_word_h, ln, rev_idx, hid_idx)
            word_hs.append(word_h)
            word_ps.append(word_p)

        sent_h, sent_p = self.forward_sent_attn(query, word_hs)
        # remove the length dimension
        sent_h = sent_h.squeeze(0)

        return (sent_h, sent_p), (word_hs, word_ps)


class SentenceAttention(ChunkedAttention):
    def __init__(self, query_size, value_size, hid_size, dropout, init_range):
        super(SentenceAttention, self).__init__(
            query_size, value_size, hid_size, init_range)

        self.word_dropout = nn.Dropout(dropout)

        self.fwd_word_rnn = nn.GRU(value_size, hid_size, bias=True)
        self.bwd_word_rnn = nn.GRU(value_size, hid_size, bias=True)
        self.word_attn = Attention(query_size, 2 * hid_size, hid_size, init_range)

        init_rnn(self.fwd_word_rnn, init_range)
        init_rnn(self.bwd_word_rnn, init_range)

    def flatten_parameters(self):
        self.fwd_word_rnn.flatten_parameters()
        self.bwd_word_rnn.flatten_parameters()

    def forward_word_attn(self, query, fwd_word_hs, bwd_word_hs, ln, rev_idx, hid_idx):
        # reverse bwd_word_h
        bwd_word_hs = bwd_word_hs.gather(0,
            rev_idx.expand(rev_idx.size(0), rev_idx.size(1), bwd_word_hs.size(2)))

        word_hs = torch.cat([fwd_word_hs, bwd_word_hs], 2)
        word_hs = self.word_dropout(word_hs)

        mask = self.make_mask([fwd_word_hs], [ln])
        word_h, word_p = self.word_attn(query, word_hs, mask)
        return word_h, word_p

    def forward_word_rnn(self, rnn, bsz, inpts, lens, rev_idxs, hid_idxs):
        hs = []
        for inpt, ln, rev_idx, hid_idx in zip(inpts, lens, rev_idxs, hid_idxs):
            h, _ = rnn(inpt, self.zero_h(bsz))
            hs.append(h)
        return hs

    def forward(self, query, fwd_inpt, fwd_len, rev_idx, hid_idx):
        # query: [batch_size, query_size]
        # fwd_inpts: [length, batch_size, value_size]
        bwd_inpts, bwd_lens = self.reverse([fwd_inpt], [fwd_len], [rev_idx])

        bsz = query.size(0)
        fwd_word_hs = self.forward_word_rnn(self.fwd_word_rnn, bsz, [fwd_inpt],
            [fwd_len], [rev_idx], [hid_idx])
        bwd_word_hs = self.forward_word_rnn(self.bwd_word_rnn, bsz, bwd_inpts,
            reversed([fwd_len]), reversed([rev_idx]), reversed([hid_idx]))

        iterator = zip(fwd_word_hs, reversed(bwd_word_hs), [fwd_len], [rev_idx], [hid_idx])
        word_hs, word_ps = [], []
        for fwd_word_h, bwd_word_h, ln, rev_idx, hid_idx in iterator:
            word_h, word_p = self.forward_word_attn(query, fwd_word_h,
                bwd_word_h, ln, rev_idx, hid_idx)
            word_hs.append(word_h)
            word_ps.append(word_p)

        return word_hs[0].squeeze(0)
