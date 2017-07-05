# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Training script. Performs supervised training of DialogModel.
"""

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
from models.dialog_model import DialogModel
import utils
from engine import Engine


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
    parser.add_argument('--nesterov', action='store_true', default=False,
        help='enable nesterov momentum')
    parser.add_argument('--clip', type=float, default=0.2,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.1,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=30,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=25,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--sel_weight', type=float, default=1.0,
        help='selection weight')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--rnn_ctx_encoder', action='store_true', default=False,
        help='wheather to use RNN for encoding the context')
    args = parser.parse_args()

    device_id = utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    corpus = data.WordCorpus(args.data, freq_cutoff=args.unk_threshold, verbose=True)
    model = DialogModel(corpus.word_dict, corpus.item_dict, corpus.context_dict,
        corpus.output_length, args, device_id)
    if device_id is not None:
        model.cuda(device_id)
    engine = Engine(model, args, device_id, verbose=True)
    train_loss, valid_loss, select_loss = engine.train(corpus)
    print('final selectppl %.3f' % np.exp(select_loss))

    utils.save_model(engine.get_model(), args.model_file)


if __name__ == '__main__':
    main()
