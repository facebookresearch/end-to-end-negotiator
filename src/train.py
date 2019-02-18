# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import time
import random
import itertools
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import utils
import models
from domain import get_domain


def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--nembed_word', type=int, default=256,
        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=64,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=256,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_cluster', type=int, default=256,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_ctx', type=int, default=64,
        help='size of the hidden state for the context module')
    parser.add_argument('--nhid_strat', type=int, default=64,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=64,
        help='size of the hidden state for the attention module')
    parser.add_argument('--nhid_sel', type=int, default=64,
        help='size of the hidden state for the selection module')
    parser.add_argument('--lr', type=float, default=20.0,
        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
        help='min threshold for learning rate annealing')
    parser.add_argument('--decay_rate', type=float,  default=9.0,
        help='decrease learning rate by this factor')
    parser.add_argument('--decay_every', type=int,  default=1,
        help='decrease learning rate after decay_every epochs')
    parser.add_argument('--momentum', type=float, default=0.0,
        help='momentum for sgd')
    parser.add_argument('--clip', type=float, default=0.2,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.1,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=30,
        help='max number of epochs')
    parser.add_argument('--num_clusters', type=int, default=50,
        help='number of clusters')
    parser.add_argument('--bsz', type=int, default=25,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--partner_ctx_weight', type=float, default=0.0,
        help='selection weight')
    parser.add_argument('--sel_weight', type=float, default=0.6,
        help='selection weight')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--prediction_model_file', type=str,  default='',
        help='path to save the prediction model')
    parser.add_argument('--selection_model_file', type=str,  default='',
        help='path to save the selection model')
    parser.add_argument('--cluster_model_file', type=str,  default='',
        help='path to save the cluster model')
    parser.add_argument('--lang_model_file', type=str,  default='',
        help='path to save the language model')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--skip_values', action='store_true', default=False,
        help='skip values in ctx encoder')
    parser.add_argument('--model_type', type=str, default='rnn_model',
        help='model type', choices=models.get_model_names())
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--clustering', action='store_true', default=False,
        help='use clustering')
    parser.add_argument('--sep_sel', action='store_true', default=False,
        help='use separate classifiers for selection')


    args = parser.parse_args()

    utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    domain = get_domain(args.domain)
    model_ty = models.get_model_type(args.model_type)
    corpus = model_ty.corpus_ty(domain, args.data, freq_cutoff=args.unk_threshold,
        verbose=True, sep_sel=args.sep_sel)
    model = model_ty(corpus.word_dict, corpus.item_dict_old,
        corpus.context_dict, corpus.count_dict, args)
    if args.cuda:
        model.cuda()
    engine = model_ty.engine_ty(model, args, verbose=True)
    train_loss, valid_loss, select_loss, extra = engine.train(corpus)

    utils.save_model(engine.get_model(), args.model_file)


if __name__ == '__main__':
    main()
