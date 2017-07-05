# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Selfplaying util.
"""

import argparse
import pdb
import re
import random

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import LstmAgent, LstmRolloutAgent, BatchedRolloutAgent
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger
from models.dialog_model import DialogModel


class SelfPlay(object):
    """Selfplay runner."""
    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        n = 0
        # goes through the list of contexes and kicks off a dialogue
        for ctxs in self.ctx_gen.iter():
            n += 1
            self.logger.dump('=' * 80)
            self.dialog.run(ctxs, self.logger)
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)


def get_agent_type(model, smart=False, fast=False):
    if isinstance(model, DialogModel):
        if smart:
            return BatchedRolloutAgent if fast else LstmRolloutAgent
        else:
            return LstmAgent
    else:
        assert False, 'unknown model type: %s' % (model)


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
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
    parser.add_argument('--fast_rollout', action='store_true', default=False,
        help='to use faster rollouts')
    parser.add_argument('--rollout_bsz', type=int, default=100,
        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    args = parser.parse_args()

    utils.set_seed(args.seed)

    alice_model = utils.load_model(args.alice_model_file)
    alice_ty = get_agent_type(alice_model, args.smart_alice, args.fast_rollout)
    alice = alice_ty(alice_model, args, name='Alice')

    bob_model = utils.load_model(args.bob_model_file)
    bob_ty = get_agent_type(bob_model, args.smart_bob, args.fast_rollout)
    bob = bob_ty(bob_model, args, name='Bob')

    dialog = Dialog([alice, bob], args)
    logger = DialogLogger(verbose=args.verbose, log_file=args.log_file)
    ctx_gen = ContextGenerator(args.context_file)

    selfplay = SelfPlay(dialog, ctx_gen, args, logger)
    selfplay.run()


if __name__ == '__main__':
    main()
