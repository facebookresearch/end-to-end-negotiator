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
from engines.rnn_engine import RnnEngine
from domain import get_domain
from models.utils import *
from models.ctx_encoder import MlpContextEncoder


class RnnModel(nn.Module):
    corpus_ty = data.WordCorpus
    engine_ty = RnnEngine
    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(RnnModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.word_encoder = nn.Embedding(len(self.word_dict), args.nembed_word)
        self.word_encoder_dropout = nn.Dropout(args.dropout)

        ctx_encoder_ty = MlpContextEncoder
        self.ctx_encoder = nn.Sequential(
            ctx_encoder_ty(len(self.context_dict), domain.input_length(), args.nembed_ctx,
                args.nhid_ctx, args.dropout, args.init_range),
            nn.Dropout(args.dropout))

        self.reader = nn.GRU(args.nhid_ctx + args.nembed_word, args.nhid_lang, bias=True)
        self.reader_dropout = nn.Dropout(args.dropout)

        self.decoder = nn.Sequential(
            nn.Linear(args.nhid_lang, args.nembed_word),
            nn.Dropout(args.dropout))

        self.writer = nn.GRUCell(
            input_size=args.nhid_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        # Tie the weights of reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.sel_rnn = nn.GRU(
            input_size=args.nhid_lang + args.nembed_word,
            hidden_size=args.nhid_attn,
            bias=True,
            bidirectional=True)
        self.sel_dropout = nn.Dropout(args.dropout)

        # Mask for disabling special tokens when generating sentences
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))

        self.sel_encoder = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn + args.nhid_ctx, args.nhid_sel),
            nn.Tanh(),
            nn.Dropout(args.dropout))
        self.attn = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn, args.nhid_attn),
            nn.Tanh(),
            torch.nn.Linear(args.nhid_attn, 1)
        )
        self.sel_decoders = nn.ModuleList()
        for i in range(domain.selection_length()):
            self.sel_decoders.append(nn.Linear(args.nhid_sel, len(self.item_dict)))

        self.init_weights()

        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

    def flatten_parameters(self):
        self.reader.flatten_parameters()
        self.sel_rnn.flatten_parameters()

    def zero_h(self, bsz, nhid=None, copies=None):
        nhid = self.args.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        h = torch.Tensor(copies, bsz, nhid).fill_(0)
        return Variable(h)

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def init_weights(self):
        #init_rnn(self.reader, self.args.init_range)
        init_cont(self.decoder, self.args.init_range)
        self.word_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)

        init_cont(self.attn, self.args.init_range)
        init_cont(self.sel_encoder, self.args.init_range)
        init_cont(self.sel_decoders, self.args.init_range)

    def read(self, inpt, lang_h, ctx_h, prefix_token='THEM:'):
        # Add a 'THEM:' token to the start of the message
        prefix = self.word2var(prefix_token).unsqueeze(0)
        inpt = torch.cat([prefix, inpt])

        inpt_emb = self.word_encoder(inpt)

        # Append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.expand(inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2))
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        # Finally read in the words
        out, lang_h = self.reader(inpt_emb, lang_h)

        return out, lang_h

    def generate_choice_logits(self, inpt, lang_h, ctx_h):
        # Run a birnn over the concatenation of the input embeddings and language model hidden states
        inpt_emb = self.word_encoder(inpt)
        inpt_emb = self.word_encoder_dropout(inpt_emb)
        h = torch.cat([lang_h.unsqueeze(1), inpt_emb], 2)

        attn_h = self.zero_h(h.size(1), self.args.nhid_attn, copies=2)
        h, _ = self.sel_rnn(h, attn_h)
        h = h.squeeze(1)

        logit = self.attn(h).squeeze(1)
        prob = F.softmax(logit).unsqueeze(1).expand_as(h)
        attn = torch.sum(torch.mul(h, prob), 0, keepdim=True)

        ctx_h = ctx_h.squeeze(1)
        h = torch.cat([attn, ctx_h], 1)
        h = self.sel_encoder.forward(h)

        logits = [decoder.forward(h).squeeze(0) for decoder in self.sel_decoders]
        return logits

    def write_batch(self, bsz, lang_h, ctx_h, temperature, max_words=100):
        eod = self.word_dict.get_idx('<selection>')

        lang_h = lang_h.squeeze(0).expand(bsz, lang_h.size(2))
        ctx_h = ctx_h.squeeze(0).expand(bsz, ctx_h.size(2))

        inpt = self.word2var('YOU:')

        outs, lang_hs = [], [lang_h.unsqueeze(0)]
        done = set()
        for _ in range(max_words):
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            lang_h = self.writer(inpt_emb, lang_h)
            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            scores.sub_(scores.max(1)[0].expand(scores.size(0), scores.size(1)))
            out = torch.multinomial(scores.exp(), 1).squeeze(1)
            outs.append(out.unsqueeze(0))
            lang_hs.append(lang_h.unsqueeze(0))
            inpt = out

            data = out.data.cpu()
            for i in range(bsz):
                if data[i] == eod:
                    done.add(i)
            if len(done) == bsz:
                break

        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h.unsqueeze(0))

        return torch.cat(outs, 0), torch.cat(lang_hs, 0)

    def write(self, lang_h, ctx_h, max_words, temperature,
            stop_tokens=data.STOP_TOKENS, resume=False):
        """
        Generate a sentence word by word and feed the output of the
        previous timestep as input to the next.
        """
        outs, logprobs, lang_hs = [], [], []
        # Remove batch dimension
        lang_h = lang_h.squeeze(1)
        ctx_h = ctx_h.squeeze(1)
        inpt = None if resume else self.word2var('YOU:')

        for _ in range(max_words):
            if inpt is not None:
                # Add the context to the word embedding
                inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
                # Update RNN state with last word
                lang_h = self.writer(inpt_emb, lang_h)
                lang_hs.append(lang_h)

            # Decode words using the inverse of the word embedding matrix
            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # Subtract constant to avoid overflows in exponentiation
            scores = scores.add(-scores.max().item()).squeeze(0)

            # Disable special tokens from being generated in a normal turns
            if not resume:
                mask = Variable(self.special_token_mask)
                scores = scores.add(mask)

            prob = F.softmax(scores, dim=0)
            logprob = F.log_softmax(scores, dim=0)

            word = prob.multinomial(1).detach()
            logprob = logprob.gather(0, word)

            logprobs.append(logprob)
            outs.append(word.view(word.size()[0], 1))

            inpt = word

            # Check if we generated an <eos> token
            if self.word_dict.get_word(word.item()) in stop_tokens:
                break

        # Update the hidden state with the <eos> token
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h)

        # Add batch dimension back
        lang_h = lang_h.unsqueeze(1)

        return logprobs, torch.cat(outs), lang_h, torch.cat(lang_hs, 0)

    def score_sent(self, sent, lang_h, ctx_h, temperature):
        score = 0
        lang_h = lang_h.squeeze(1)
        ctx_h = ctx_h.squeeze(1)
        inpt = self.word2var('YOU:')
        lang_hs = []

        for word in sent:
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            lang_h = self.writer(inpt_emb, lang_h)
            lang_hs.append(lang_h)

            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            scores = scores.add(-scores.max().data[0]).squeeze(0)

            mask = Variable(self.special_token_mask)
            scores = scores.add(mask)

            logprob = F.log_softmax(scores)
            score += logprob[word[0]].data[0]
            inpt = Variable(word)

        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h)

        lang_h = lang_h.unsqueeze(1)

        return score, lang_h, torch.cat(lang_hs)

    def forward_context(self, ctx):
        ctx_h = self.ctx_encoder(ctx).unsqueeze(0)
        return ctx_h

    def forward_lm(self, inpt_emb, lang_h, ctx_h):
        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.narrow(0, ctx_h.size(0) - 1, 1).expand(
            inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2))
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        lang_hs, _ = self.reader(inpt_emb, lang_h)
        lang_hs = self.reader_dropout(lang_hs)

        decoded = self.decoder(lang_hs.view(-1, lang_hs.size(2)))
        out = F.linear(decoded, self.word_encoder.weight)

        return out, lang_hs

    def forward_selection(self, inpt_emb, lang_h, ctx_h):
        # run a birnn over the concatenation of the input embeddings and language model hidden states
        h = torch.cat([lang_h, inpt_emb], 2)

        attn_h = self.zero_h(h.size(1), self.args.nhid_attn, copies=2)
        h, _ = self.sel_rnn(h, attn_h)
        h = self.sel_dropout(h)

        h = h.transpose(0, 1).contiguous()
        logit = self.attn(h.view(-1, 2 * self.args.nhid_attn)).view(h.size(0), h.size(1))
        prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(h)
        attn = torch.sum(torch.mul(h, prob), 1, keepdim=True).transpose(0, 1).contiguous()

        h = torch.cat([attn, ctx_h], 2).squeeze(0)
        h = self.sel_encoder.forward(h)

        outs = [decoder.forward(h) for decoder in self.sel_decoders]
        out = torch.cat(outs, 0)
        return out

    def forward(self, inpt, ctx):
        ctx_h = self.forward_context(ctx)
        lang_h = self.zero_h(ctx_h.size(1), self.args.nhid_lang)

        inpt_emb = self.word_encoder(inpt)
        inpt_emb = self.word_encoder_dropout(inpt_emb)

        out, lang_hs = self.forward_lm(inpt_emb, lang_h, ctx_h)
        sel_out = self.forward_selection(inpt_emb, lang_hs, ctx_h)

        return out, sel_out
