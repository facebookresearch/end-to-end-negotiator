# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to compute average rank for both supervised models
and models with planning.
"""

import argparse
import sys
import time
import random
import itertools
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
from agent import LstmAgent
from dialog import DialogLogger
import utils
from domain import get_domain


TAGS = ['YOU:', 'THEM:']


def read_dataset(file_name):
    """A helper function that reads the dataset and finds out all the unique sentences."""
    lines = data.read_lines(file_name)
    dataset = []
    all_sents = set()

    for line in lines:
        tokens = line.split(' ')
        ctx = data.get_tag(tokens, 'input')
        sents, sent = [], []
        you = None
        for t in data.get_tag(tokens, 'dialogue'):
            if t in TAGS:
                if you is not None:
                    sents.append((sent, you))
                    if you:
                        all_sents.add(' '.join(sent))
                    sent = []
                you = t == 'YOU:'
            else:
                assert you is not None
                sent.append(t)
                if t == '<selection>':
                    break

        if len(sent) > 0:
            sents.append((sent, you))
            if you:
                all_sents.add(' '.join(sent))

        dataset.append((ctx, sents))

    sents = [sent.split(' ') for sent in all_sents]
    random.shuffle(dataset)
    return dataset, sents


def rollout(sent, ai, domain, temperature):
    enc = ai._encode(sent, ai.model.word_dict)
    _, lang_h, lang_hs = ai.model.score_sent(enc, ai.lang_h, ai.ctx_h, temperature)

    is_selection = len(sent) == 1 and sent[0] == '<selection>'

    score = 0
    for _ in range(5):
        combined_lang_hs = ai.lang_hs + [lang_hs]
        combined_words = ai.words + [ai.model.word2var('YOU:'), Variable(enc)]

        if not is_selection:
            # complete the conversation with rollout_length samples
            _, rollout, _, rollout_lang_hs = ai.model.write(
                lang_h, ai.ctx_h, 100, temperature,
                stop_tokens=['<selection>'], resume=True)
            combined_lang_hs += [rollout_lang_hs]
            combined_words += [rollout]

        # choose items
        rollout_score = None

        combined_lang_hs = torch.cat(combined_lang_hs)
        combined_words = torch.cat(combined_words)

        rollout_choice, _, p_agree = ai._choose(combined_lang_hs, combined_words, sample=False)
        rollout_score = domain.score(ai.context, rollout_choice)
        score += p_agree * rollout_score

    return score


def likelihood(sent, ai, domain, temperature):
    """Computes likelihood of a given sentence according the giving model."""
    enc = ai._encode(sent, ai.model.word_dict)
    score, _, _= ai.model.score_sent(enc, ai.lang_h, ai.ctx_h, temperature)
    return score


def compute_rank(target, sents, ai, domain, temperature, score_func):
    """Computes rank of the target sentence.

    Basically find a position in the sorted list of all seen sentences.
    """
    scores = []
    # score each unique sentence
    for sent in sents:
        score = score_func(sent, ai, domain, temperature)
        scores.append((score, sent))
    scores = sorted(scores, key=lambda x: -x[0])

    # score the target sentence
    target_score = score_func(target, ai, domain, temperature)

    # find the position of the target sentence in the sorted list of all the senteces
    for rank, (score, _) in enumerate(scores):
        if target_score > score:
            return rank + 1
    return len(scores) + 1


def main():
    parser = argparse.ArgumentParser(description='Negotiator')
    parser.add_argument('--dataset', type=str, default='./data/negotiate/val.txt',
        help='location of the dataset')
    parser.add_argument('--model_file', type=str,
        help='model file')
    parser.add_argument('--smart_ai', action='store_true', default=False,
        help='to use rollouts')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--log_file', type=str, default='',
        help='log file')
    args = parser.parse_args()

    utils.set_seed(args.seed)

    model = utils.load_model(args.model_file)
    ai = LstmAgent(model, args)
    logger = DialogLogger(verbose=True, log_file=args.log_file)
    domain = get_domain(args.domain)

    score_func = rollout if args.smart_ai else likelihood

    dataset, sents = read_dataset(args.dataset)
    ranks, n, k = 0, 0, 0
    for ctx, dialog in dataset:
        start_time = time.time()
        # start new conversation
        ai.feed_context(ctx)
        for sent, you in dialog:
            if you:
                # if it is your turn to say, take the target word and compute its rank
                rank = compute_rank(sent, sents, ai, domain, args.temperature, score_func)
                # compute lang_h for the groundtruth sentence
                enc = ai._encode(sent, ai.model.word_dict)
                _, ai.lang_h, lang_hs = ai.model.score_sent(enc, ai.lang_h, ai.ctx_h, args.temperature)
                # save hidden states and the utterance
                ai.lang_hs.append(lang_hs)
                ai.words.append(ai.model.word2var('YOU:'))
                ai.words.append(Variable(enc))
                ranks += rank
                n += 1
            else:
                ai.read(sent)
        k += 1
        time_elapsed = time.time() - start_time
        logger.dump('dialogue %d | avg rank %.3f | raw %d/%d | time %.3f' % (k, 1. * ranks / n, ranks, n, time_elapsed))

    logger.dump('final avg rank %.3f' % (1. * ranks / n))


if __name__ == '__main__':
    main()
