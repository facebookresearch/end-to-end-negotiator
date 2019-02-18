# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

from engines import EngineBase, Criterion
import utils


class LatentClusteringEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(LatentClusteringEngine, self).__init__(model, args, verbose)
        self.crit = nn.CrossEntropyLoss(reduction='sum')
        self.kldiv = nn.KLDivLoss(reduction='sum')
        self.cluster_crit = nn.NLLLoss(reduction='sum')
        self.sel_crit = Criterion(
            self.model.item_dict,
            bad_toks=['<disconnect>', '<disagree>'],
            reduction='mean' if args.sep_sel else 'none')

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

    def _make_sel_tgt_probs(self, inpts, lens, rev_idxs, hid_idxs, ctx):
        sel_tgt_probs = []
        for i in range(len(inpts)):
            sel_prob = self.sel_model(inpts[:i], lens[:i], rev_idxs[:i], hid_idxs[:i], ctx)
            sel_tgt_probs.append(F.softmax(sel_prob.detach(), dim=1))
        return sel_tgt_probs

    def _append_pad(self, inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs):
        bsz = inpts[0].size(1)
        pad = torch.Tensor(bsz).fill_(self.model.word_dict.get_idx('<pad>')).long()
        inpts.append(Variable(pad.unsqueeze(0)))
        tgts.append(Variable(pad))
        sel_tgt_probs.append(sel_tgt_probs[-1].clone())
        lens.append(torch.Tensor(bsz).cpu().fill_(0).long())
        rev_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        hid_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        return inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs

    def _forward(self, batch, norm_lang=True):
        ctx, _, inpts, lens, tgts, sel_tgt, rev_idxs, hid_idxs, cnt = batch
        ctx = Variable(ctx)
        cnt = Variable(cnt)
        inpts = [Variable(inpt) for inpt in inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]
        sel_tgt_probs = self._make_sel_tgt_probs(inpts, lens, rev_idxs, hid_idxs, ctx)
        sel_tgt = Variable(sel_tgt)

        inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs = self._append_pad(
            inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs)

        outs, sel_outs, z_probs, z_tgts, stats = self.model(
            inpts, tgts, sel_tgt_probs, hid_idxs, ctx, cnt)

        lang_loss, n = 0, 0
        for out, tgt, ln in zip(outs, tgts[1:], lens[1:]):
            lang_loss += self.crit(out, tgt)
            n += ln.sum()
        if norm_lang:
            lang_loss /= n

        z_loss, n = 0, 0
        for z_prob, z_tgt in zip(z_probs, z_tgts):
            z_loss += self.cluster_crit(z_prob.log(), z_tgt)
            n += z_tgt.size(0)
        z_loss /= n

        kldiv_loss, n = 0, 0
        for sel_out, sel_tgt_prob in zip(sel_outs, sel_tgt_probs[1:]):
            kldiv_loss += self.kldiv(F.log_softmax(sel_out, dim=1), sel_tgt_prob)
            n += sel_out.size(0)
        kldiv_loss /= n

        sel_loss = self.sel_crit(sel_outs[-1], sel_tgt)

        return lang_loss, sel_loss, kldiv_loss, z_loss, stats

    def combine_loss(self, lang_loss, select_loss):
        return lang_loss + select_loss

    def train_batch(self, batch):
        lang_loss, sel_loss, kldiv_loss, z_loss, stats = self._forward(
            batch)

        loss = lang_loss + sel_loss + kldiv_loss + z_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()

        return loss.item(), stats

    def valid_batch(self, batch):
        with torch.no_grad():
            lang_loss, sel_loss, kldiv_loss, z_loss, stats = self._forward(
                batch, norm_lang=False)

        return lang_loss.item(), sel_loss.item(), 0, stats

    def train_pass(self, trainset):
        self.model.train()

        total_loss = 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        total_enc_entropy = 0
        total_enc_max_prob = 0
        total_enc_top3_prob = 0

        start_time = time.time()

        for batch in trainset:
            self.t += 1
            loss, (entropy, max_prob, top3_prob, enc_entropy, enc_max_prob, enc_top3_prob) = self.train_batch(batch)

            if self.args.visual and self.t % 100 == 0:
                self.model_plot.update(self.t)

            total_loss += loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob
            total_enc_entropy += enc_entropy
            total_enc_max_prob += enc_max_prob
            total_enc_top3_prob += enc_top3_prob

        total_loss /= len(trainset)
        total_entropy /= len(trainset)
        total_max_prob /= len(trainset)
        total_top3_prob /= len(trainset)
        total_enc_entropy /= len(trainset)
        total_enc_max_prob /= len(trainset)
        total_enc_top3_prob /= len(trainset)

        time_elapsed = time.time() - start_time
        print('| train | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f ' % (
            total_entropy, total_max_prob, total_top3_prob))
        print('| train | enc avg entropy %.3f | enc avg max prob %.3f | enc avg top3 prob %.3f ' % (
            total_enc_entropy, total_enc_max_prob, total_enc_top3_prob))

        return total_loss, time_elapsed

    def valid_pass(self, validset, validset_stats):
        self.model.eval()

        total_valid_loss, total_select_loss, total_partner_ctx_loss = 0, 0, 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        total_enc_entropy = 0
        total_enc_max_prob = 0
        total_enc_top3_prob = 0
        for batch in validset:
            valid_loss, select_loss, partner_ctx_loss, (entropy, max_prob, top3_prob, enc_entropy, enc_max_prob, enc_top3_prob) = self.valid_batch(batch)
            total_valid_loss += valid_loss
            total_select_loss += select_loss
            total_partner_ctx_loss += partner_ctx_loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob
            total_enc_entropy += enc_entropy
            total_enc_max_prob += enc_max_prob
            total_enc_top3_prob += enc_top3_prob

        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        total_valid_loss /= validset_stats['nonpadn']
        total_select_loss /= len(validset)
        total_partner_ctx_loss /= len(validset)
        #total_future_loss /= len(validset)
        total_entropy /= len(validset)
        total_max_prob /= len(validset)
        total_top3_prob /= len(validset)
        total_enc_entropy /= len(validset)
        total_enc_max_prob /= len(validset)
        total_enc_top3_prob /= len(validset)

        print('| valid | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f ' % (
            total_entropy, total_max_prob, total_top3_prob))
        print('| valid | enc avg entropy %.3f | enc avg max prob %.3f | enc avg top3 prob %.3f ' % (
            total_enc_entropy, total_enc_max_prob, total_enc_top3_prob))

        extra = {
            'entropy': total_entropy,
            'avg_max_prob': total_max_prob,
            'avg_top3_prob': total_top3_prob,
        }

        return total_valid_loss, total_select_loss, total_partner_ctx_loss, extra


class LatentClusteringPredictionEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(LatentClusteringPredictionEngine, self).__init__(model, args, verbose)
        self.crit = nn.CrossEntropyLoss(reduction='sum')
        self.model.train()

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

    def _make_sel_tgt_probs(self, inpts, lens, rev_idxs, hid_idxs, ctx):
        sel_tgt_probs = []
        for i in range(len(inpts)):
            sel_prob = self.sel_model(inpts[:i], lens[:i], rev_idxs[:i], hid_idxs[:i], ctx)
            sel_tgt_probs.append(F.softmax(sel_prob.detach(), dim=1))
        return sel_tgt_probs

    def _forward(self, batch):
        ctx, _, inpts, lens, tgts, sel_tgt, rev_idxs, hid_idxs, cnt = batch
        ctx = Variable(ctx)
        cnt = Variable(cnt)
        inpts = [Variable(inpt) for inpt in inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]
        sel_tgt_probs = self._make_sel_tgt_probs(inpts, lens, rev_idxs, hid_idxs, ctx)

        losses, stats = self.model.forward(inpts, tgts, hid_idxs, ctx, cnt)

        return losses, stats, lens

    def train_batch(self, batch):
        losses, stats, lens = self._forward(batch)

        lang_loss, n = 0, 0
        for l, ln in zip(losses, lens):
            lang_loss += l
            n += ln.sum()
        lang_loss /= n

        loss = lang_loss
        self.opt.zero_grad()
        loss.backward()

        # don't update clusters
        self.model.latent_bottleneck.zero_grad()
        # don't update language model
        self.model.lang_model.zero_grad()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return lang_loss.item(), stats

    def valid_batch(self, batch):
        with torch.no_grad():
            losses, stats, lens = self._forward(batch)

        loss = 0
        for l in losses:
            loss += l

        return loss.item(), 0, 0, stats

    def train_pass(self, trainset):
        self.model.train()

        total_loss = 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        start_time = time.time()

        for batch in trainset:
            self.t += 1
            loss, (entropy, max_prob, top3_prob) = self.train_batch(batch)

            if self.args.visual and self.t % 100 == 0:
                self.model_plot.update(self.t)

            total_loss += loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob

        total_loss /= len(trainset)
        total_entropy /= len(trainset)
        total_max_prob /= len(trainset)
        total_top3_prob /= len(trainset)
        time_elapsed = time.time() - start_time
        print('| train | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f' % (
            total_entropy, total_max_prob, total_top3_prob))

        return total_loss, time_elapsed

    def valid_pass(self, validset, validset_stats):
        self.model.eval()

        total_valid_loss, total_select_loss, total_partner_ctx_loss = 0, 0, 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        for batch in validset:
            valid_loss, select_loss, partner_ctx_loss, (entropy, max_prob, top3_prob) = self.valid_batch(batch)
            total_valid_loss += valid_loss
            total_select_loss += select_loss
            total_partner_ctx_loss += partner_ctx_loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob

        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        total_valid_loss /= validset_stats['nonpadn']
        total_select_loss /= len(validset)
        total_partner_ctx_loss /= len(validset)
        #total_future_loss /= len(validset)
        total_entropy /= len(validset)
        total_max_prob /= len(validset)
        total_top3_prob /= len(validset)
        print('| valid | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f' % (
            total_entropy, total_max_prob, total_top3_prob))

        extra = {
            'entropy': total_entropy,
            'avg_max_prob': total_max_prob,
            'avg_top3_prob': total_top3_prob,
        }

        return total_valid_loss, total_select_loss, total_partner_ctx_loss, extra


class LatentClusteringLanguageEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(LatentClusteringLanguageEngine, self).__init__(model, args, verbose)
        self.crit = nn.CrossEntropyLoss(reduction='sum')
        self.cluster_crit = nn.NLLLoss(reduction='sum')

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

    def _make_sel_tgt_probs(self, inpts, lens, rev_idxs, hid_idxs, ctx):
        sel_tgt_probs = []
        for i in range(len(inpts)):
            sel_prob = self.sel_model(inpts[:i], lens[:i], rev_idxs[:i], hid_idxs[:i], ctx)
            sel_tgt_probs.append(F.softmax(sel_prob.detach(), dim=1))
        return sel_tgt_probs

    def _append_pad(self, inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs):
        bsz = inpts[0].size(1)
        pad = torch.Tensor(bsz).fill_(self.model.word_dict.get_idx('<pad>')).long()
        inpts.append(Variable(pad.unsqueeze(0)))
        tgts.append(Variable(pad))
        sel_tgt_probs.append(sel_tgt_probs[-1].clone())
        lens.append(torch.Tensor(bsz).cpu().fill_(0).long())
        rev_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        hid_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        return inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs

    def _forward(self, model, batch, sep_sel=False, norm_lang=False):
        ctx, _, inpts, lens, tgts, sel_tgt, rev_idxs, hid_idxs, cnt = batch
        ctx = Variable(ctx)
        cnt = Variable(cnt)
        inpts = [Variable(inpt) for inpt in inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]
        sel_tgt_probs = self._make_sel_tgt_probs(inpts, lens, rev_idxs, hid_idxs, ctx)

        inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs = self._append_pad(
            inpts, tgts, sel_tgt_probs, lens, rev_idxs, hid_idxs)

        outs = model(inpts, tgts, sel_tgt_probs, hid_idxs, ctx, cnt)

        lang_loss, n = 0, 0
        for out, tgt, ln in zip(outs, tgts, lens):
            lang_loss += self.crit(out, tgt)
            n += ln.sum()
        if norm_lang:
            lang_loss /= n

        return lang_loss

    def train_batch(self, batch):
        lang_loss = self._forward(
            self.model, batch, sep_sel=self.args.sep_sel, norm_lang=True)

        loss = lang_loss

        self.opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return lang_loss.item()

    def valid_batch(self, batch):
        with torch.no_grad():
            lang_loss = self._forward(
                self.model, batch, sep_sel=self.args.sep_sel, norm_lang=False)

        return lang_loss.item(), 0, 0


class BaselineClusteringEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(BaselineClusteringEngine, self).__init__(model, args, verbose)
        self.model.train()

    def _forward(self, batch):
        ctx, _, inpts, lens, tgts, sel_tgt, rev_idxs, hid_idxs, cnt = batch
        ctx = Variable(ctx)
        cnt = Variable(cnt)
        inpts = [Variable(inpt) for inpt in inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]

        losses, stats = self.model.forward(inpts, tgts, hid_idxs, ctx, cnt)

        return losses, stats, lens

    def train_batch(self, batch):
        losses, stats, lens = self._forward(batch)

        lang_loss, n = 0, 0
        for l, ln in zip(losses, lens):
            lang_loss += l
            n += ln.sum()
        lang_loss /= n

        loss = lang_loss
        self.opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return lang_loss.item(), stats

    def valid_batch(self, batch):
        with torch.no_grad():
            losses, stats, lens = self._forward(batch)

        loss = 0
        for l in losses:
            loss += l

        return loss.item(), 0, 0, stats

    def train_pass(self, trainset):
        self.model.train()

        total_loss = 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        start_time = time.time()

        for batch in trainset:
            self.t += 1
            loss, (entropy, max_prob, top3_prob) = self.train_batch(batch)

            if self.args.visual and self.t % 100 == 0:
                self.model_plot.update(self.t)

            total_loss += loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob

        total_loss /= len(trainset)
        total_entropy /= len(trainset)
        total_max_prob /= len(trainset)
        total_top3_prob /= len(trainset)
        time_elapsed = time.time() - start_time
        print('| train | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f' % (
            total_entropy, total_max_prob, total_top3_prob))

        return total_loss, time_elapsed

    def valid_pass(self, validset, validset_stats):
        self.model.eval()

        total_valid_loss, total_select_loss, total_partner_ctx_loss = 0, 0, 0
        total_entropy = 0
        total_max_prob = 0
        total_top3_prob = 0
        for batch in validset:
            valid_loss, select_loss, partner_ctx_loss, (entropy, max_prob, top3_prob) = self.valid_batch(batch)
            total_valid_loss += valid_loss
            total_select_loss += select_loss
            total_partner_ctx_loss += partner_ctx_loss
            total_entropy += entropy
            total_max_prob += max_prob
            total_top3_prob += top3_prob

        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        total_valid_loss /= validset_stats['nonpadn']
        total_select_loss /= len(validset)
        total_partner_ctx_loss /= len(validset)
        #total_future_loss /= len(validset)
        total_entropy /= len(validset)
        total_max_prob /= len(validset)
        total_top3_prob /= len(validset)
        print('| valid | avg entropy %.3f | avg max prob %.3f | avg top3 prob %.3f' % (
            total_entropy, total_max_prob, total_top3_prob))

        extra = {
            'entropy': total_entropy,
            'avg_max_prob': total_max_prob,
            'avg_top3_prob': total_top3_prob,
        }

        return total_valid_loss, total_select_loss, total_partner_ctx_loss, extra
