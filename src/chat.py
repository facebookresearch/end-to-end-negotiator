# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import domain

import utils
from utils import ContextGenerator, ManualContextGenerator
from agent import RnnAgent, HumanAgent, RnnRolloutAgent, HierarchicalAgent
from dialog import Dialog, DialogLogger


class Chat(object):
    def __init__(self, dialog, ctx_gen, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.logger = logger if logger else DialogLogger()

    def run(self):
        self.logger.dump('Welcome to our Chatroulette!')
        for dialog_id in itertools.count():
            self.logger.dump('=' * 80)
            self.logger.dump('Dialog %d' % dialog_id)
            self.logger.dump('-' * 80)
            ctxs = self.ctx_gen.sample()
            self.dialog.run(ctxs, self.logger)
            self.logger.dump('=' * 80)
            self.logger.dump('')


def main():
    parser = argparse.ArgumentParser(description='chat utility')
    parser.add_argument('--model_file', type=str,
        help='model file')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--context_file', type=str, default='',
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--num_types', type=int, default=3,
        help='number of object types')
    parser.add_argument('--num_objects', type=int, default=6,
        help='total number of objects')
    parser.add_argument('--max_score', type=int, default=10,
        help='max score per object')
    parser.add_argument('--score_threshold', type=int, default=6,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--smart_ai', action='store_true', default=False,
        help='make AI smart again')
    parser.add_argument('--ai_starts', action='store_true', default=False,
        help='allow AI to start the dialog')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    args = parser.parse_args()

    utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    human = HumanAgent(domain.get_domain(args.domain))

    alice_ty = RnnRolloutAgent if args.smart_ai else HierarchicalAgent
    ai = alice_ty(utils.load_model(args.model_file), args)


    agents = [ai, human] if args.ai_starts else [human, ai]

    dialog = Dialog(agents, args)
    logger = DialogLogger(verbose=True)
    if args.context_file == '':
        ctx_gen = ManualContextGenerator(args.num_types, args.num_objects, args.max_score)
    else:
        ctx_gen = ContextGenerator(args.context_file)

    chat = Chat(dialog, ctx_gen, logger)
    chat.run()


if __name__ == '__main__':
    main()
