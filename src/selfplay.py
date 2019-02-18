# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pdb
import re
import random

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import *
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger
from models.rnn_model import RnnModel
from models.latent_clustering_model import LatentClusteringPredictionModel, BaselineClusteringModel
import domain


class SelfPlay(object):
    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        n = 0
        for ctxs in self.ctx_gen.iter():
            n += 1
            if self.args.smart_alice and n > 1000:
                break
            self.logger.dump('=' * 80)
            self.dialog.run(ctxs, self.logger)
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)


def get_agent_type(model, smart=False):
    if isinstance(model, LatentClusteringPredictionModel):
        if smart:
            return LatentClusteringRolloutAgent
        else:
            return LatentClusteringAgent
    elif isinstance(model, RnnModel):
        if smart:
            return RnnRolloutAgent
        else:
            return RnnAgent
    elif isinstance(model, BaselineClusteringModel):
        if smart:
            return BaselineClusteringRolloutAgent
        else:
            return BaselineClusteringAgent
    else:
        assert False, 'unknown model type: %s' % (model)


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--alice_forward_model_file', type=str,
        help='Alice forward model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--score_threshold', type=int, default=6,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--max_turns', type=int, default=20,
        help='maximum number of turns in a dialog')
    parser.add_argument('--log_file', type=str, default='',
        help='log successful dialogs to file for training')
    parser.add_argument('--smart_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--diverse_alice', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--rollout_bsz', type=int, default=3,
        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--selection_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--rollout_model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--diverse_bob', action='store_true', default=False,
        help='make Alice smart again')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
        help='eps greedy')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--validate', action='store_true', default=False,
        help='plot graphs')

    args = parser.parse_args()

    utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    alice_model = utils.load_model(args.alice_model_file)
    alice_ty = get_agent_type(alice_model, args.smart_alice)
    alice = alice_ty(alice_model, args, name='Alice', train=False, diverse=args.diverse_alice)
    alice.vis = args.visual

    bob_model = utils.load_model(args.bob_model_file)
    bob_ty = get_agent_type(bob_model, args.smart_bob)
    bob = bob_ty(bob_model, args, name='Bob', train=False, diverse=args.diverse_bob)

    bob.vis = False

    dialog = Dialog([alice, bob], args)
    logger = DialogLogger(verbose=args.verbose, log_file=args.log_file)
    ctx_gen = ContextGenerator(args.context_file)

    selfplay = SelfPlay(dialog, ctx_gen, args, logger)
    selfplay.run()


if __name__ == '__main__':
    main()
