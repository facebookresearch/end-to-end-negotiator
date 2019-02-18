# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import time
import itertools
import sys
import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import vis


class Criterion(object):
    """Weighted CrossEntropyLoss."""
    def __init__(self, dictionary, device_id=None, bad_toks=[], reduction='mean'):
        w = torch.Tensor(len(dictionary)).fill_(1)
        for tok in bad_toks:
            w[dictionary.get_idx(tok)] = 0.0
        if device_id is not None:
            w = w.cuda(device_id)
        # https://pytorch.org/docs/stable/nn.html
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def __call__(self, out, tgt):
        return self.crit(out, tgt)


class EngineBase(object):
    """Base class for training engine."""
    def __init__(self, model, args, verbose=False):
        self.model = model
        self.args = args
        self.verbose = verbose
        self.opt = self.make_opt(self.args.lr)
        self.crit = Criterion(self.model.word_dict)
        self.sel_crit = Criterion(
            self.model.item_dict, bad_toks=['<disconnect>', '<disagree>'])
        if self.args.visual:
            self.model_plot = vis.ModulePlot(self.model, plot_weight=True, plot_grad=False)
            self.loss_plot = vis.Plot(['train', 'valid', 'valid_select'],
                'loss', 'loss', 'epoch', running_n=1, write_to_file=False)
            self.ppl_plot = vis.Plot(['train', 'valid', 'valid_select'],
                'perplexity', 'ppl', 'epoch', running_n=1, write_to_file=False)

    def make_opt(self, lr):
        return optim.RMSprop(
            self.model.parameters(),
            lr=lr,
            momentum=self.args.momentum)

    def get_model(self):
        return self.model

    def train_batch(self, batch):
        pass

    def valid_batch(self, batch):
        pass

    def train_pass(self, trainset):
        self.model.train()

        total_loss = 0
        start_time = time.time()

        for batch in trainset:
            self.t += 1
            loss = self.train_batch(batch)

            if self.args.visual and self.t % 100 == 0:
                self.model_plot.update(self.t)

            total_loss += loss

        total_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_loss, time_elapsed

    def valid_pass(self, validset, validset_stats):
        self.model.eval()

        total_valid_loss, total_select_loss, total_partner_ctx_loss = 0, 0, 0
        for batch in validset:
            valid_loss, select_loss, partner_ctx_loss = self.valid_batch(batch)
            total_valid_loss += valid_loss
            total_select_loss += select_loss
            total_partner_ctx_loss += partner_ctx_loss

        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        total_valid_loss /= validset_stats['nonpadn']
        total_select_loss /= len(validset)
        total_partner_ctx_loss /= len(validset)
        #total_future_loss /= len(validset)
        return total_valid_loss, total_select_loss, total_partner_ctx_loss, {}

    def iter(self, epoch, lr, traindata, validdata):
        trainset, _ = traindata
        validset, validset_stats = validdata

        train_loss, train_time = self.train_pass(trainset)
        valid_loss, valid_select_loss, valid_partner_ctx_loss, extra = \
            self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('| epoch %03d | trainloss %.3f | trainppl %.3f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_loss, np.exp(train_loss), train_time, lr))
            print('| epoch %03d | validloss %.3f | validppl %.3f' % (
                epoch, valid_loss, np.exp(valid_loss)))
            print('| epoch %03d | validselectloss %.3f | validselectppl %.3f' % (
                epoch, valid_select_loss, np.exp(valid_select_loss)))
            if self.model.args.partner_ctx_weight != 0:
                print('| epoch %03d | validpartnerctxloss %.3f | validpartnerctxppl %.3f' % (
                    epoch, valid_partner_ctx_loss, np.exp(valid_partner_ctx_loss)))

        if self.args.visual:
            self.loss_plot.update('train', epoch, train_loss)
            self.loss_plot.update('valid', epoch, valid_loss)
            self.loss_plot.update('valid_select', epoch, valid_select_loss)
            self.ppl_plot.update('train', epoch, np.exp(train_loss))
            self.ppl_plot.update('valid', epoch, np.exp(valid_loss))
            self.ppl_plot.update('valid_select', epoch, np.exp(valid_select_loss))

        return train_loss, valid_loss, valid_select_loss, extra

    def combine_loss(self, lang_loss, select_loss):
        return lang_loss + select_loss

    def train(self, corpus):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        lr = self.args.lr
        last_decay_epoch = 0
        self.t = 0

        validdata = corpus.valid_dataset(self.args.bsz)
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            _, valid_loss, valid_select_loss, extra = self.iter(epoch, lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_loss, valid_select_loss)
            if combined_valid_loss < best_combined_valid_loss:
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        if self.verbose:
            print('| start annealing | best combined loss %.3f | best combined ppl %.3f' % (
                best_combined_valid_loss, np.exp(best_combined_valid_loss)))

        self.model = best_model
        for epoch in range(self.args.max_epoch + 1, 100):
            if epoch - last_decay_epoch >= self.args.decay_every:
                last_decay_epoch = epoch
                lr /= self.args.decay_rate
                if lr < self.args.min_lr:
                    break
                self.opt = self.make_opt(lr)

            traindata = corpus.train_dataset(self.args.bsz)
            train_loss, valid_loss, valid_select_loss, extra = self.iter(
                epoch, lr, traindata, validdata)

        return train_loss, valid_loss, valid_select_loss, extra
