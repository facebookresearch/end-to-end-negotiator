# Copyright 2019-present, Facebook, Inc.
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
import utils
from domain import get_domain
from models.ctx_encoder import MlpContextEncoder
from models.utils import *



from engines.latent_clustering_engine import LatentClusteringEngine, LatentClusteringPredictionEngine
from engines.latent_clustering_engine import LatentClusteringLanguageEngine, BaselineClusteringEngine


class SimpleSeparateSelectionModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(SimpleSeparateSelectionModule, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout))

        self.decoders = nn.ModuleList()
        for i in range(6):
            self.decoders.append(nn.Linear(hidden_size, output_size))

        # init
        init_cont(self.encoder, args.init_range)
        init_cont(self.decoders, args.init_range)

    def flatten_parameters(self):
        pass

    def forward(self, h):
        h = self.encoder(h)
        outs = [decoder(h) for decoder in self.decoders]
        out = torch.cat(outs, 1).view(-1, self.output_size)
        return out


class RecurrentUnit(nn.Module):
    def __init__(self, input_size, hidden_size, args):
        super(RecurrentUnit, self).__init__()

        self.x2h = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh())

        self.cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bias=True)

        # init
        init_cont(self.x2h, args.init_range)
        init_rnn_cell(self.cell, args.init_range)

    def forward(self, x, h):
        x_h = self.x2h(x)
        h = self.cell(x_h, h)
        return h


class ShardedLatentBottleneckModule(nn.Module):
    def __init__(self, num_shards, num_clusters, input_size, output_size, args):
        super(ShardedLatentBottleneckModule, self).__init__()

        self.num_clusters = num_clusters
        self.input_size = input_size
        self.output_size = output_size

        self.latent_vars = nn.Embedding(num_shards, num_clusters * output_size)
        self.weight = nn.Embedding(num_shards, input_size * num_clusters)
        self.bias = nn.Embedding(num_shards, num_clusters)

        # init
        self.latent_vars.weight.data.uniform_(-args.init_range, args.init_range)
        self.weight.weight.data.uniform_(-args.init_range, args.init_range)
        self.bias.weight.data.uniform_(-args.init_range, args.init_range)

    def zero_grad(self):
        if self.latent_vars.weight.grad is not None:
            self.latent_vars.weight.grad.data.zero_()

    def select(self, shard, idx):
        lat_var = self.latent_vars(shard)
        lat_var = lat_var.view(-1, self.num_clusters, self.output_size)
        out = []
        for i in range(lat_var.size(0)):
            out.append(lat_var[i][idx[i]].unsqueeze(0))
        out = torch.cat(out, 0)
        return out

    def select_shard(self, shard):
        lat_var = self.latent_vars(shard)
        lat_var = lat_var.view(-1, self.num_clusters, self.output_size)
        return lat_var

    def forward(self, shard, key):
        # find corresponding weights
        lat_var = self.latent_vars(shard)
        w = self.weight(shard)
        b = self.bias(shard)

        # unpack
        lat_var = lat_var.view(-1, self.num_clusters, self.output_size)
        w = w.view(-1, self.input_size, self.num_clusters)

        # dot product
        logit = torch.bmm(key.unsqueeze(1),  w).squeeze(1) + b
        prob = F.softmax(logit, dim=1)

        # weighted sum
        lat_h = torch.bmm(prob.unsqueeze(1), lat_var).squeeze(1)

        return lat_h, prob


class LatentClusteringModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = LatentClusteringEngine

    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(LatentClusteringModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.ctx_encoder = MlpContextEncoder(len(self.context_dict), domain.input_length(),
            args.nembed_ctx, args.nhid_ctx, args.dropout, args.init_range, args.skip_values)

        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        self.hid2output = nn.Sequential(
            nn.Linear(args.nhid_lang, args.nembed_word),
            nn.Dropout(args.dropout))

        self.mem2input = nn.Linear(args.nhid_lang, args.nembed_word)

        self.encoder = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.embed2hid = nn.Sequential(
            nn.Linear(args.nhid_lang + args.nhid_lang + args.nhid_ctx, args.nhid_cluster),
            nn.Tanh())

        self.decoder_reader = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.decoder_writer = nn.GRUCell(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        # tie the weights between reader and writer
        self.decoder_writer.weight_ih = self.decoder_reader.weight_ih_l0
        self.decoder_writer.weight_hh = self.decoder_reader.weight_hh_l0
        self.decoder_writer.bias_ih = self.decoder_reader.bias_ih_l0
        self.decoder_writer.bias_hh = self.decoder_reader.bias_hh_l0

        self.latent_bottleneck = ShardedLatentBottleneckModule(
            num_shards=len(count_dict),
            num_clusters=args.num_clusters,
            input_size=args.nhid_lang,
            output_size=args.nhid_cluster,
            args=args)

        self.memory = nn.GRUCell(
            input_size=args.nhid_cluster,
            hidden_size=args.nhid_lang,
            bias=True)

        self.dropout = nn.Dropout(args.dropout)

        self.selection = SimpleSeparateSelectionModule(
            input_size=args.nhid_cluster,
            hidden_size=args.nhid_sel,
            output_size=len(item_dict),
            args=args)

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.encoder, args.init_range)
        init_rnn(self.decoder_reader, args.init_range)
        init_rnn_cell(self.memory, args.init_range)
        init_linear(self.mem2input, args.init_range)
        init_cont(self.hid2output, args.init_range)
        init_cont(self.embed2hid, args.init_range)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder_reader.flatten_parameters()
        self.selection.flatten_parameters()

    def embed_sentence(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def encode_sentence(self, inpt_emb, hid_idx):
        def _encode(e, hid_idx=None):
            h = self._zero(1, e.size(1), self.args.nhid_lang)
            # encode
            hs, h = self.encoder(e, h)
            if hid_idx is not None:
                # extract correct hidden states to avoid padding
                h = hs.gather(0, hid_idx.expand(hid_idx.size(0), hid_idx.size(1), hs.size(2)))
            # remove temporal dimension
            h = h.squeeze(0)
            return h

        # encode
        sent_h = _encode(inpt_emb, hid_idx)
        turn_h = _encode(inpt_emb.narrow(0, 0, 1))

        enc_h = torch.cat([sent_h, turn_h], 1)
        return enc_h

    def decode_sentence(self, inpt_emb, mem_h):
        # embed memory state
        mem_emb = self.mem2input(mem_h)

        # init hidden state with memory state
        lang_h = self._zero(inpt_emb.size(1), self.args.nhid_lang)
        lang_h = self.decoder_writer(mem_emb, lang_h)

        # add temporal dimension
        lang_h = lang_h.unsqueeze(0)
        # run decoder
        dec_hs, _ = self.decoder_reader(inpt_emb, lang_h)
        return dec_hs

    def unembed_sentence(self, dec_hs):
        # convert to the embed space
        out_emb = self.hid2output(dec_hs.view(-1, dec_hs.size(2)))
        # unembed
        out = F.linear(out_emb, self.word_embed.weight)
        return out

    def forward_encoder(self, ctx_h, inpt, hid_idx):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # encode sentence and turn
        sent_h = self.encode_sentence(inpt_emb, hid_idx)
        # combine sent_h and ctx_h
        enc_h = self.embed2hid(torch.cat([sent_h, ctx_h], 1))
        return enc_h

    def forward_decoder(self, inpt, mem_h):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # run decoder
        dec_hs = self.decode_sentence(inpt_emb, mem_h)
        # unembed
        out = self.unembed_sentence(dec_hs)
        return out

    def forward_e_step(self, z_prob, mem_h, inpt, tgt, sel_tgt_prob, cnt):
        # select all clusters for each shard
        lat_h = self.latent_bottleneck.select_shard(cnt)

        inpt_len, batch_size = inpt.size()
        num_clusters = lat_h.size(1)

        # duplicate mem_h
        mem_h = mem_h.unsqueeze(1).expand_as(lat_h).contiguous()

        # duplicate inpt_emb
        inpt = inpt.unsqueeze(2).expand(
            inpt_len, batch_size, lat_h.size(1)).contiguous()
        tgt = tgt.unsqueeze(1).expand(tgt.size(0), lat_h.size(1)).contiguous()
        sel_tgt_prob = sel_tgt_prob.view(batch_size, 1, -1, sel_tgt_prob.size(1))
        sel_tgt_prob = sel_tgt_prob.expand(
            batch_size, num_clusters, sel_tgt_prob.size(2), sel_tgt_prob.size(3)).contiguous()

        # batch together all the clusters
        lat_h = lat_h.view(-1, lat_h.size(2))
        mem_h = mem_h.view(-1, mem_h.size(2))
        inpt = inpt.view(inpt_len, -1)
        tgt = tgt.view(-1, 1)
        #sel_tgt_prob = sel_tgt_prob.view(-1, sel_tgt_prob.size(3))

        # run through memory
        mem_h = self.memory(lat_h, mem_h)

        # run decoding
        out = self.forward_decoder(inpt, mem_h)

        # find log(p(x_{t+1} | z_t) * p(s_t | z_t) * p(z_t | x_t)) =
        # = log p(x_{t+1} | z_t) + log p(s_t | z_t) + log p(z_t | x_t)

        logprob = F.log_softmax(out, dim=1)
        cross_entropy = logprob.gather(1, tgt)
        # break batch out
        cross_entropy = cross_entropy.view(inpt_len, batch_size, num_clusters).sum(0)

        # find KL-div for selection
        sel_out = self.selection(mem_h)
        sel_logprob = F.log_softmax(sel_out, dim=1)
        sel_logprob = sel_logprob.view_as(sel_tgt_prob)
        sel_cross_entropy = (sel_tgt_prob * sel_logprob).sum(3).sum(2)

        z_logprob = z_prob.log()

        # q(z_{t+1}), posterior distribution over z
        q_z = cross_entropy + sel_cross_entropy + z_logprob

        # get cluster ids
        _, z = q_z.max(1)
        z = z.detach()
        return z, q_z

    def forward(self, inpts, tgts, sel_tgt_probs, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        mem_h = self._zero(bsz, self.args.nhid_lang)

        outs, sel_outs = [], []
        z_probs, z_tgts = [], []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        total_enc_entropy, total_enc_max_prob, total_enc_top3_prob = 0, 0, 0
        # skip 1st sentence
        for i in range(len(inpts) - 1):
            # run encoder
            enc_h = self.forward_encoder(ctx_h, inpts[i], hid_idxs[i])

            # estimate p(z|x)
            _, z_prob = self.latent_bottleneck(cnt, enc_h)
            z_probs.append(z_prob)

            # E-step, estimate z for i+1 sentence
            z, q_z = self.forward_e_step(z_prob, mem_h,
                inpts[i + 1], tgts[i + 1], sel_tgt_probs[i + 1], cnt)
            z_tgts.append(z)

            # select clusters for z
            lat_h = self.latent_bottleneck.select(cnt, z)

            # run through memory
            mem_h = self.memory(lat_h, mem_h)

            # compute selection
            sel_out = self.selection(mem_h)
            sel_outs.append(sel_out)

            # decode next sentence
            out = self.forward_decoder(inpts[i + 1], mem_h)
            outs.append(out)

            # book keeping
            total_entropy += -(F.softmax(q_z, dim=1) * F.log_softmax(q_z, dim=1)).sum().item()
            q_z = F.softmax(q_z, dim=1)
            total_max_prob += q_z.max(1, keepdim=True)[0].sum().item()
            total_top3_prob += torch.topk(q_z, 3)[0].sum().item()

            total_enc_entropy += -(z_prob * z_prob.log()).sum().item()
            total_enc_max_prob += z_prob.max(1, keepdim=True)[0].sum().item()
            total_enc_top3_prob += torch.topk(z_prob, 3)[0].sum().item()

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        total_enc_entropy /= len(inpts) * bsz
        total_enc_max_prob /= len(inpts) * bsz
        total_enc_top3_prob /= len(inpts) * bsz

        stats = (total_entropy, total_max_prob, total_top3_prob,
            total_enc_entropy, total_enc_max_prob, total_enc_top3_prob)

        return outs, sel_outs, z_probs, z_tgts, stats


class LatentClusteringPredictionModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = LatentClusteringPredictionEngine

    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(LatentClusteringPredictionModel, self).__init__()

        self.lang_model = utils.load_model(args.lang_model_file)
        self.lang_model.eval()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.ctx_encoder = MlpContextEncoder(len(self.context_dict), domain.input_length(),
            args.nembed_ctx, args.nhid_ctx, args.dropout, args.init_range, False)

        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        self.encoder = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.embed2hid = nn.Sequential(
            nn.Linear(args.nhid_lang + args.nhid_lang + args.nhid_ctx, self.args.nhid_lang),
            nn.Tanh())

        self.latent_bottleneck = ShardedLatentBottleneckModule(
            num_shards=len(count_dict),
            num_clusters=self.lang_model.cluster_model.args.num_clusters,
            input_size=args.nhid_lang,
            output_size=self.lang_model.cluster_model.args.nhid_cluster,
            args=args)

        # copy lat vars from the cluster model
        self.latent_bottleneck.latent_vars.weight.data.copy_(
            self.lang_model.cluster_model.latent_bottleneck.latent_vars.weight.data)

        self.memory = RecurrentUnit(
            input_size=args.nhid_lang,
            hidden_size=self.lang_model.cluster_model.args.nhid_cluster,
            args=args)

        self.dropout = nn.Dropout(args.dropout)

        self.kldiv = nn.KLDivLoss(reduction='sum')

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.encoder, args.init_range)
        init_cont(self.embed2hid, args.init_range)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.lang_model.flatten_parameters()

    def clear_weights(self):
        modules = [
            self.ctx_encoder,
            self.word_embed,
            self.encoder,
            self.embed2hid,
            self.latent_bottleneck,
            self.memory
        ]
        # clear all the parameters
        for module in modules:
            for param in module.parameters():
                param.data.uniform_(-self.args.init_range, self.args.init_range)
        # copy lat vars from the cluster model
        self.latent_bottleneck.latent_vars.weight.data.copy_(
            self.lang_model.cluster_model.latent_bottleneck.latent_vars.weight.data)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def embed_sentence(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def encode_sentence(self, inpt_emb, hid_idx):
        def _encode(e, hid_idx=None):
            h = self._zero(1, e.size(1), self.args.nhid_lang)
            # encode
            hs, h = self.encoder(e, h)
            if hid_idx is not None:
                # extract correct hidden states to avoid padding
                h = hs.gather(0, hid_idx.expand(hid_idx.size(0), hid_idx.size(1), hs.size(2)))
            # remove temporal dimension
            h = h.squeeze(0)
            return h

        # encode
        sent_h = _encode(inpt_emb, hid_idx)
        turn_h = _encode(inpt_emb.narrow(0, 0, 1))

        enc_h = torch.cat([sent_h, turn_h], 1)
        return enc_h

    def forward_encoder(self, ctx_h, inpt=None, hid_idx=None):
        if inpt is not None:
            # embed
            inpt_emb = self.embed_sentence(inpt)
            # encode sentence and turn
            sent_h = self.encode_sentence(inpt_emb, hid_idx)
            # combine sent_h and ctx_h
        else:
            # start with empty
            sent_h = self._zero(ctx_h.size(0), 2 * self.args.nhid_lang)
        # combine sent_h and ctx_h
        enc_h = self.embed2hid(torch.cat([sent_h, ctx_h], 1))
        return enc_h

    def forward_memory(self, ctx_h, mem_h=None, inpt=None, hid_idx=None):
        # run encoder
        enc_h = self.forward_encoder(ctx_h, inpt, hid_idx)
        # init mem_h if None
        if mem_h is None:
            mem_h = self._zero(ctx_h.size(0), self.lang_model.cluster_model.args.nhid_cluster)
        # update memory
        mem_h = self.memory(enc_h, mem_h)
        return mem_h

    def forward_prediction(self, cnt, mem_h, sample=False):
        # estimate q(z)
        _, q_z = self.latent_bottleneck(cnt, mem_h)
        # either sample or take max
        if sample:
            z = q_z.multinomial().detach()
            log_q_z = q_z.log().gather(1, z)
        else:
            _, z = q_z.max(1)
            log_q_z = None

        # select centroides
        lat_h = self.latent_bottleneck.select(cnt, z)
        return z, lat_h, log_q_z

    def forward_prediction_multi(self, cnt, mem_h, num_samples, sample=False):
        # estimate q(z)
        _, q_z = self.latent_bottleneck(cnt, mem_h)
        # either sample or take max
        if sample:
            z = q_z.multinomial(num_samples, replacement=True).detach()
        else:
            _, z = torch.topk(q_z, num_samples)

        # select centroides
        lat_h = self.latent_bottleneck.select(cnt, z)
        return z, lat_h

    def forward_marginal_loss(self, q_z, lang_enc_h, inpt, tgt, cnt):
        # select all clusters for each shard
        cluster_lat_h = self.latent_bottleneck.select_shard(cnt)

        inpt_len, batch_size = inpt.size()
        num_clusters = cluster_lat_h.size(1)

        # duplicate enc_h
        lang_enc_h = lang_enc_h.unsqueeze(1).expand_as(cluster_lat_h).contiguous()

        # combine enc_h and cluster_lat_h
        cond_h = torch.cat([lang_enc_h, cluster_lat_h], 2)

        # duplicate inpt_emb
        inpt = inpt.unsqueeze(2).expand(
            inpt_len, batch_size, cluster_lat_h.size(1)).contiguous()
        tgt = tgt.unsqueeze(1).expand(tgt.size(0), cluster_lat_h.size(1)).contiguous()

        # batch together all the clusters
        cluster_lat_h = cluster_lat_h.view(-1, cluster_lat_h.size(2))
        cond_h = cond_h.view(-1, cond_h.size(2))
        inpt = inpt.view(inpt_len, -1)
        tgt = tgt.view(-1, 1)

        # run decoding
        out = self.lang_model.forward_decoder(inpt, cond_h)

        logprob = F.log_softmax(out, dim=1)
        cross_entropy = logprob.gather(1, tgt)
        cross_entropy = cross_entropy.view(inpt_len, batch_size, num_clusters)
        cross_entropy = cross_entropy.sum(0)

        logit = cross_entropy + q_z.log()
        mx = logit.max(1, keepdim=True)[0]
        loss = mx.squeeze(1) + torch.log(torch.exp(logit.sub(mx)).sum(1))
        loss = -loss.sum()
        return loss

    def forward_validation(self, inpts, tgts, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        # init encoders
        enc_h = self.embed2hid(
            torch.cat([self._zero(ctx_h.size(0), 2 * self.args.nhid_lang), ctx_h], 1))

        lang_enc_h = self._zero(ctx_h.size(0), self.lang_model.args.nhid_lang)

        outs, losses = [], []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        for inpt, tgt, hid_idx in zip(inpts, tgts, hid_idxs):
            next_lang_enc_h = self.lang_model.forward_encoder(inpt, hid_idx, lang_enc_h)

            _, z_prob = self.lang_model.latent_bottleneck(cnt, next_lang_enc_h)
            z = self.lang_model.forward_e_step(z_prob, lang_enc_h, inpt, tgt, cnt)

            lat_h = self.latent_bottleneck.select(cnt, z)

            cond_h = torch.cat([lang_enc_h, lat_h], 1)

            out = self.lang_model.forward_decoder(inpt, cond_h)
            outs.append(out)

            lang_enc_h = next_lang_enc_h

            # bookkeeping
            total_entropy += -(z_prob * z_prob.log()).sum().data[0]
            total_max_prob += z_prob.max(1, keepdim=True)[0].sum().data[0]
            total_top3_prob += torch.topk(z_prob, 3)[0].sum().data[0]

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        stats = (total_entropy, total_max_prob, total_top3_prob)

        return outs, losses, stats

    def forward_validation_marginal(self, inpts, tgts, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        # init encoders
        enc_h = self.embed2hid(
            torch.cat([self._zero(ctx_h.size(0), 2 * self.args.nhid_lang), ctx_h], 1))

        lang_enc_h = self._zero(ctx_h.size(0), self.lang_model.args.nhid_lang)

        outs, losses = [], []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        for inpt, tgt, hid_idx in zip(inpts, tgts, hid_idxs):
            next_lang_enc_h = self.lang_model.forward_encoder(inpt, hid_idx, lang_enc_h)

            _, z_prob = self.lang_model.latent_bottleneck(cnt, next_lang_enc_h)
            z = self.lang_model.forward_e_step(z_prob, lang_enc_h, inpt, tgt, cnt)
            one_hot = Variable(torch.Tensor(z_prob.size()).zero_().scatter_(1, z.unsqueeze(1).data, 1))

            loss = self.forward_marginal_loss(one_hot, lang_enc_h, inpt, tgt, cnt)
            losses.append(loss)

            lang_enc_h = next_lang_enc_h

            # bookkeeping
            total_entropy += -(z_prob * z_prob.log()).sum().data[0]
            total_max_prob += z_prob.max(1, keepdim=True)[0].sum().data[0]
            total_top3_prob += torch.topk(z_prob, 3)[0].sum().data[0]

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        stats = (total_entropy, total_max_prob, total_top3_prob)

        return outs, losses, stats

    def forward(self, inpts, tgts, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        # init lang encode from zero
        lang_enc_h = self._zero(bsz, self.lang_model.args.nhid_lang)

        # init memory
        mem_h = self.forward_memory(ctx_h, mem_h=None, inpt=None)

        losses = []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        for inpt, tgt, hid_idx in zip(inpts, tgts, hid_idxs):
            # predictic q(z)
            _, q_z = self.latent_bottleneck(cnt, mem_h)

            loss = self.forward_marginal_loss(q_z, lang_enc_h, inpt, tgt, cnt)
            losses.append(loss)

            # update language encoder
            lang_enc_h = self.lang_model.forward_encoder(inpt, hid_idx, lang_enc_h)
            # update memory
            mem_h = self.forward_memory(ctx_h, mem_h, inpt, hid_idx)

            # bookkeeping
            total_entropy += -(q_z * q_z.log()).sum().item()
            total_max_prob += q_z.max(1, keepdim=True)[0].sum().item()
            total_top3_prob += torch.topk(q_z, 3)[0].sum().item()

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        stats = (total_entropy, total_max_prob, total_top3_prob)

        return losses, stats

    def forward_kldiv(self, inpts, tgts, sel_tgt_probs, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        cluster_ctx_h = self.lang_model.cluster_model.ctx_encoder(ctx)
        cluster_mem_h = self._zero(cluster_ctx_h.size(0), self.lang_model.cluster_model.args.nhid_lang)

        # init encoders
        enc_h = self.embed2hid(
            torch.cat([self._zero(ctx_h.size(0), 2 * self.args.nhid_lang), ctx_h], 1))
        lang_enc_h = self._zero(ctx_h.size(0), self.lang_model.args.nhid_lang)

        # init memory
        mem_h = self._zero(ctx_h.size(0), self.lang_model.cluster_model.args.nhid_cluster)

        losses, kldivs = [], []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        for i in range(len(inpts) - 1):
            # update memory
            mem_h = self.memory(enc_h, mem_h)

            # predictic q(z)
            _, q_z = self.latent_bottleneck(cnt, mem_h)

            loss = self.forward_marginal_loss(q_z, lang_enc_h, inpts[i], tgts[i], cnt)
            losses.append(loss)

            # estimate p(z|x)
            cluster_enc_h = self.lang_model.cluster_model.forward_encoder(
                cluster_ctx_h, inpts[i], hid_idxs[i])
            _, z_prob = self.lang_model.cluster_model.latent_bottleneck(cnt, cluster_enc_h)
            # run e-step to estimate z, q_z
            z, q_t_tgt = self.lang_model.cluster_model.forward_e_step(z_prob, cluster_mem_h,
                inpts[i + 1], tgts[i + 1], sel_tgt_probs[i + 1], cnt)
            cluster_lat_h = self.lang_model.cluster_model.latent_bottleneck.select(cnt, z)
            q_t_tgt = q_t_tgt.detach()
            kldivs.append(self.kldiv(q_z.log(), F.softmax(q_t_tgt)))
            # update memory
            cluster_mem_h = self.lang_model.cluster_model.memory(cluster_lat_h, cluster_mem_h)

            # update encoders
            enc_h = self.forward_encoder(ctx_h, inpts[i], hid_idxs[i])
            lang_enc_h = self.lang_model.forward_encoder(inpts[i], hid_idxs[i], lang_enc_h)

            # bookkeeping
            total_entropy += -(q_z * q_z.log()).sum().data[0]
            total_max_prob += q_z.max(1, keepdim=True)[0].sum().data[0]
            total_top3_prob += torch.topk(q_z, 3)[0].sum().data[0]

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        stats = (total_entropy, total_max_prob, total_top3_prob)

        return losses, kldivs, stats

    def read(self, inpt, lang_enc_h, mem_h, ctx_h):
        # make hid_idx, since it's just one sentence take full len
        hid_idx = Variable(torch.Tensor(1, 1, 1).fill_(inpt.size(0) - 1).long())
        # update memory
        mem_h = self.forward_memory(ctx_h, mem_h, inpt, hid_idx)
        # update language encoder
        lang_enc_h = self.lang_model.forward_encoder(inpt, hid_idx, lang_enc_h)
        return lang_enc_h, mem_h

    def write(self, lang_enc_h, lat_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):
        # construct condition
        cond_h = torch.cat([lang_enc_h, lat_h], 1)
        # send to language model
        out, logprobs = self.lang_model.write(cond_h, max_words, temperature,
            start_token, stop_tokens)
        return out, logprobs


class LatentClusteringLanguageModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = LatentClusteringLanguageEngine

    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(LatentClusteringLanguageModel, self).__init__()

        self.cluster_model = utils.load_model(args.cluster_model_file)
        self.cluster_model.eval()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        self.encoder = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.hid2output = nn.Sequential(
            nn.Linear(args.nhid_lang, args.nembed_word),
            nn.Dropout(args.dropout))

        self.cond2input = nn.Linear(
            args.nhid_lang + self.cluster_model.args.nhid_cluster,
            args.nembed_word)

        self.decoder_reader = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.decoder_writer = nn.GRUCell(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        # tie the weights between reader and writer
        self.decoder_writer.weight_ih = self.decoder_reader.weight_ih_l0
        self.decoder_writer.weight_hh = self.decoder_reader.weight_hh_l0
        self.decoder_writer.bias_ih = self.decoder_reader.bias_ih_l0
        self.decoder_writer.bias_hh = self.decoder_reader.bias_hh_l0

        self.dropout = nn.Dropout(args.dropout)

        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.decoder_reader, args.init_range)
        init_linear(self.cond2input, args.init_range)
        init_cont(self.hid2output, args.init_range)
        init_rnn(self.encoder, args.init_range)

    def flatten_parameters(self):
        self.decoder_reader.flatten_parameters()
        self.encoder.flatten_parameters()
        self.cluster_model.flatten_parameters()

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

    def embed_sentence(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def encode_sentence(self, inpt_emb, hid_idx, enc_h):
        enc_h = enc_h.unsqueeze(0)
        # encode
        hs, _ = self.encoder(inpt_emb, enc_h)
        # extract correct hidden states to avoid padding
        enc_h = hs.gather(0, hid_idx.expand(hid_idx.size(0), hid_idx.size(1), hs.size(2)))
        # remove temporal dimension
        enc_h = enc_h.squeeze(0)
        return enc_h

    def decode_sentence(self, inpt_emb, cond_h):
        # embed latent state
        cond_emb = self.cond2input(cond_h)

        # init hidden state with memory state
        lang_h = self._zero(inpt_emb.size(1), self.args.nhid_lang)
        lang_h = self.decoder_writer(cond_emb, lang_h)

        # add temporal dimension
        lang_h = lang_h.unsqueeze(0) # run decoder
        dec_hs, _ = self.decoder_reader(inpt_emb, lang_h)
        return dec_hs

    def unembed_sentence(self, dec_hs):
        # convert to the embed space
        out_emb = self.hid2output(dec_hs.view(-1, dec_hs.size(2)))
        # unembed
        out = F.linear(out_emb, self.word_embed.weight)
        return out

    def forward_encoder(self, inpt, hid_idx, enc_h):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # run encoder
        enc_h = self.encode_sentence(inpt_emb, hid_idx, enc_h)
        return enc_h

    def forward_decoder(self, inpt, cond_h):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # run decoder
        dec_hs = self.decode_sentence(inpt_emb, cond_h)
        # unembed
        out = self.unembed_sentence(dec_hs)
        return out

    def forward(self, inpts, tgts, sel_tgt_probs, hid_idxs, ctx, cnt):
        cluster_ctx_h = self.cluster_model.ctx_encoder(ctx)
        cluster_mem_h = self._zero(cluster_ctx_h.size(0), self.cluster_model.args.nhid_lang)

        enc_h = self._zero(ctx.size(1), self.args.nhid_lang)

        outs = []
        for i in range(len(inpts) - 1):
            # estimate p(z|x)
            cluster_enc_h = self.cluster_model.forward_encoder(
                cluster_ctx_h, inpts[i], hid_idxs[i])
            _, z_prob = self.cluster_model.latent_bottleneck(cnt, cluster_enc_h)
            # run e-step to estimate z, q_z
            z, _ = self.cluster_model.forward_e_step(z_prob, cluster_mem_h,
                inpts[i + 1], tgts[i + 1], sel_tgt_probs[i + 1], cnt)
            # select centroids
            cluster_lat_h = self.cluster_model.latent_bottleneck.select(cnt, z)
            # update cluster model memory
            cluster_mem_h = self.cluster_model.memory(cluster_lat_h, cluster_mem_h)
            # detach
            cluster_lat_h = cluster_lat_h.detach()

            # combine enc_h  and cluster_lat_h
            cond_h = torch.cat([enc_h, cluster_lat_h], 1)

            # decode
            out = self.forward_decoder(inpts[i], cond_h)
            outs.append(out)

            # run encoder
            enc_h = self.forward_encoder(inpts[i], hid_idxs[i], enc_h)

        return outs

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def write(self, cond_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):

        # init lang_h
        cond_emb = self.cond2input(cond_h)
        lang_h = self._zero(cond_emb.size(0), self.args.nhid_lang)
        lang_h = self.decoder_writer(cond_emb, lang_h)

        # autoregress starting from start_token
        inpt = self.word2var(start_token)

        outs = [inpt.unsqueeze(0)]
        logprobs = []
        for _ in range(max_words):
            # embed
            inpt_emb = self.embed_sentence(inpt)

            # decode next word
            lang_h = self.decoder_writer(inpt_emb, lang_h)

            if self.word_dict.get_word(inpt.data[0]) in stop_tokens:
                break

            # unembed
            out = self.unembed_sentence(lang_h.unsqueeze(0))

            scores = out.div(temperature)
            scores = scores.sub(scores.max().item()).squeeze(0)

            mask = Variable(self.special_token_mask)
            scores = scores.add(mask)

            prob = F.softmax(scores, dim=0)
            logprob = F.log_softmax(scores, dim=0)
            inpt = prob.multinomial(1).detach()
            outs.append(inpt.unsqueeze(0))
            logprob = logprob.gather(0, inpt)
            logprobs.append(logprob)

        out = torch.cat(outs, 0)
        return out, logprobs


class BaselineClusteringModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = BaselineClusteringEngine

    def __init__(self, word_dict, item_dict, context_dict, count_dict, args):
        super(BaselineClusteringModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.count_dict = count_dict
        self.args = args

        self.ctx_encoder = MlpContextEncoder(len(self.context_dict), domain.input_length(),
            args.nembed_ctx, args.nhid_lang, args.dropout, args.init_range, False)

        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        self.encoder = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.latent_bottleneck = ShardedLatentBottleneckModule(
            num_shards=len(count_dict),
            num_clusters=self.args.num_clusters,
            input_size=args.nhid_lang,
            output_size=self.args.nhid_cluster,
            args=args)

        self.dropout = nn.Dropout(args.dropout)

        self.decoder_reader = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.decoder_writer = nn.GRUCell(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.cond2input = nn.Linear(
            args.nhid_cluster,
            args.nembed_word)

        self.hid2output = nn.Sequential(
            nn.Linear(args.nhid_lang, args.nembed_word),
            nn.Dropout(args.dropout))

        self.memory = RecurrentUnit(
            input_size=args.nhid_lang,
            hidden_size=args.nhid_lang,
            args=args)

        # tie the weights between reader and writer
        self.decoder_writer.weight_ih = self.decoder_reader.weight_ih_l0
        self.decoder_writer.weight_hh = self.decoder_reader.weight_hh_l0
        self.decoder_writer.bias_ih = self.decoder_reader.bias_ih_l0
        self.decoder_writer.bias_hh = self.decoder_reader.bias_hh_l0

        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.encoder, args.init_range)
        init_rnn(self.decoder_reader, args.init_range)
        init_linear(self.cond2input, args.init_range)
        init_cont(self.hid2output, args.init_range)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder_reader.flatten_parameters()

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def embed_sentence(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def decode_sentence(self, inpt_emb, cond_h):
        # embed latent state
        cond_emb = self.cond2input(cond_h)

        # init hidden state with memory state
        lang_h = self._zero(inpt_emb.size(1), self.args.nhid_lang)
        lang_h = self.decoder_writer(cond_emb, lang_h)

        # add temporal dimension
        lang_h = lang_h.unsqueeze(0) # run decoder
        dec_hs, _ = self.decoder_reader(inpt_emb, lang_h)
        return dec_hs

    def unembed_sentence(self, dec_hs):
        # convert to the embed space
        out_emb = self.hid2output(dec_hs.view(-1, dec_hs.size(2)))
        # unembed
        out = F.linear(out_emb, self.word_embed.weight)
        return out

    def forward_encoder(self, inpt, hid_idx):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # encode sentence
        enc_h = self._zero(1, inpt_emb.size(1), self.args.nhid_lang)
        out, _ = self.encoder(inpt_emb, enc_h)
        # extract correct hidden states to avoid padding
        enc_h = out.gather(0, hid_idx.expand(hid_idx.size(0), hid_idx.size(1), out.size(2)))
        return enc_h.squeeze(0)

    def init_memory(self, ctx_h):
        mem_h = self._zero(ctx_h.size(0), self.args.nhid_lang)
        # input ctx_h
        mem_h = self.memory(ctx_h, mem_h)
        return mem_h

    def forward_memory(self, mem_h, inpt, hid_idx):
        # run encoder
        enc_h = self.forward_encoder(inpt, hid_idx)
        # update memory
        mem_h = self.memory(enc_h, mem_h)
        return mem_h

    def forward_decoder(self, inpt, cond_h):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # run decoder
        dec_hs = self.decode_sentence(inpt_emb, cond_h)
        # unembed
        out = self.unembed_sentence(dec_hs)
        return out

    def forward_marginal_loss(self, p_z, inpt, tgt, cnt):
        # select all clusters for each shard
        lat_h = self.latent_bottleneck.select_shard(cnt)

        inpt_len, batch_size = inpt.size()
        num_clusters = lat_h.size(1)

        # duplicate inpt_emb
        inpt = inpt.unsqueeze(2).expand(
            inpt_len, batch_size, lat_h.size(1)).contiguous()
        tgt = tgt.unsqueeze(1).expand(tgt.size(0), lat_h.size(1)).contiguous()

        # batch together all the clusters
        lat_h = lat_h.view(-1, lat_h.size(2))
        inpt = inpt.view(inpt_len, -1)
        tgt = tgt.view(-1, 1)

        # run decoding
        out = self.forward_decoder(inpt, lat_h)

        logprob = F.log_softmax(out)
        cross_entropy = logprob.gather(1, tgt)
        cross_entropy = cross_entropy.view(inpt_len, batch_size, num_clusters)
        cross_entropy = cross_entropy.sum(0)

        logit = cross_entropy + p_z.log()
        mx = logit.max(1, keepdim=True)[0]
        loss = mx.squeeze(1) + torch.log(torch.exp(logit.sub(mx)).sum(1))
        loss = -loss.sum()
        return loss

    def forward(self, inpts, tgts, hid_idxs, ctx, cnt):
        ctx_h = self.ctx_encoder(ctx)
        bsz = ctx_h.size(0)

        mem_h = self.init_memory(ctx_h)

        losses = []
        total_entropy, total_max_prob, total_top3_prob = 0, 0, 0
        for inpt, tgt, hid_idx in zip(inpts, tgts, hid_idxs):
            _, p_z = self.latent_bottleneck(cnt, mem_h)

            loss = self.forward_marginal_loss(p_z, inpt, tgt, cnt)
            losses.append(loss)

            # update encoder
            mem_h = self.forward_memory(mem_h, inpt, hid_idx)

            # bookkeeping
            total_entropy += -(p_z * p_z.log()).sum().data[0]
            total_max_prob += p_z.max(1, keepdim=True)[0].sum().data[0]
            total_top3_prob += torch.topk(p_z, 3)[0].sum().data[0]

        total_entropy /= len(inpts) * bsz
        total_max_prob /= len(inpts) * bsz
        total_top3_prob /= len(inpts) * bsz
        stats = (total_entropy, total_max_prob, total_top3_prob)

        return losses, stats

    def read(self, inpt, mem_h):
        # make hid_idx, since it's just one sentence take full len
        hid_idx = Variable(torch.Tensor(1, 1, 1).fill_(inpt.size(0) - 1).long())
        # update memory
        mem_h = self.forward_memory(mem_h, inpt, hid_idx)
        return mem_h

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def write(self, cond_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):

        # init lang_h
        cond_emb = self.cond2input(cond_h)
        lang_h = self._zero(cond_emb.size(0), self.args.nhid_lang)
        lang_h = self.decoder_writer(cond_emb, lang_h)

        # autoregress starting from start_token
        inpt = self.word2var(start_token)

        outs = [inpt.unsqueeze(0)]
        for _ in range(max_words):
            # embed
            inpt_emb = self.embed_sentence(inpt)

            # decode next word
            lang_h = self.decoder_writer(inpt_emb, lang_h)

            if self.word_dict.get_word(inpt.data[0]) in stop_tokens:
                break

            # unembed
            out = self.unembed_sentence(lang_h.unsqueeze(0))

            scores = out.div(temperature)
            scores = scores.sub(scores.max().data[0]).squeeze(0)

            mask = Variable(self.special_token_mask)
            scores = scores.add(mask)

            prob = F.softmax(scores)
            inpt = prob.multinomial().detach()
            outs.append(inpt.unsqueeze(0))

        out = torch.cat(outs, 0)
        return out
