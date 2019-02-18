# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Training script. Performs supervised training of DialogModel.
"""

import argparse
import itertools
import logging
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import time

# local imports
import config
import data
from engine import Engine
from models.dialog_model import DialogModel
import utils

logging.basicConfig(format=config.log_format, level=config.log_level)

def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data', type=str, default=config.data_dir,
        help='location of the data corpus')
    parser.add_argument('--nembed_word', type=int, default=config.nembed_word,
        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=config.nembed_ctx,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=config.nhid_lang,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_ctx', type=int, default=config.nhid_ctx,
        help='size of the hidden state for the context module')
    parser.add_argument('--nhid_strat', type=int, default=config.nhid_strat,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=config.nhid_attn,
        help='size of the hidden state for the attention module')
    parser.add_argument('--nhid_sel', type=int, default=config.nhid_sel,
        help='size of the hidden state for the selection module')
    parser.add_argument('--lr', type=float, default=config.lr,
        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=config.min_lr,
        help='min threshold for learning rate annealing')
    parser.add_argument('--decay_rate', type=float,  default=config.decay_rate,
        help='decrease learning rate by this factor')
    parser.add_argument('--decay_every', type=int,  default=config.decay_every,
        help='decrease learning rate after decay_every epochs')
    parser.add_argument('--momentum', type=float, default=config.momentum,
        help='momentum for sgd')
    parser.add_argument('--nesterov', action='store_true', default=config.nesterov,
        help='enable nesterov momentum')
    parser.add_argument('--clip', type=float, default=config.clip,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=config.dropout,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=config.init_range,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=config.max_epoch,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=config.bsz,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=config.unk_threshold,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=config.temperature,
        help='temperature')
    parser.add_argument('--sel_weight', type=float, default=config.sel_weight,
        help='selection weight')
    parser.add_argument('--seed', type=int, default=config.seed,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=config.cuda,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--visual', action='store_true', default=config.plot_graphs,
        help='plot graphs')
    parser.add_argument('--domain', type=str, default=config.domain,
        help='domain for the dialogue')
    parser.add_argument('--rnn_ctx_encoder', action='store_true', default=config.rnn_ctx_encoder,
        help='whether to use RNN for encoding the context')
    args = parser.parse_args()

    device_id = utils.use_cuda(args.cuda)
    logging.info('Starting training using pytorch version:%s' % (str(torch.__version__)))
    logging.info('CUDA is %s' % ('enabled. Using device_id:' + str(device_id) + ' version:' \
        + str(torch.version.cuda) + ' on gpu:' + torch.cuda.get_device_name(0) if args.cuda else 'disabled'))
    utils.set_seed(args.seed)

    logging.info('Building word corpus, requiring minimum word frequency of %d for dictionary' % (args.unk_threshold))
    corpus = data.WordCorpus(args.data, freq_cutoff=args.unk_threshold, verbose=True)

    logging.info('Building RNN-based dialogue model from word corpus')
    model = DialogModel(corpus.word_dict, corpus.item_dict, corpus.context_dict,
        corpus.output_length, args, device_id)
    if device_id is not None:
        model.cuda(device_id)
    
    engine = Engine(model, args, device_id, verbose=True)
    logging.info('Training model')
    train_loss, valid_loss, select_loss = engine.train(corpus)
    logging.info('final select_ppl %.3f' % np.exp(select_loss))

    utils.save_model(engine.get_model(), args.model_file)


if __name__ == '__main__':
    main()
