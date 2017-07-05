# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to evaluate selfplay.
It computes agreement rate, average score and Pareto optimality.
"""

import argparse
import sys
import time
import random
import itertools
import re
import pdb

import numpy as np

import data
import utils
from domain import get_domain


def parse_line(line, domain):
    # skip the 'debug:' token
    tokens = line.split(' ')[1:]
    context = tokens[:2 * domain.input_length()]
    choice_str = tokens[-domain.selection_length():]

    cnts, vals = domain.parse_context(context)
    picks = []
    for i, c in enumerate(choice_str[:domain.selection_length() // 2]):
        if c in ('<disconnect>', '<no_agreement>'):
            picks.append(-1)
        else:
            idx, pick = domain.parse_choice(c)
            assert idx == i
            picks.append(pick)

    return cnts, vals, picks


def parse_log(file_name, domain):
    """Parse the log file produced by selfplay.
    See the format of that log file to get more details.
    """
    dataset, current = [], []
    for line in data.read_lines(file_name):
        if line.startswith('debug:'):
            cnts, vals, picks = parse_line(line, domain)
            current.append((cnts, vals, picks))
        if len(current) == 2:
            # validate that the counts match
            cnts1, vals1, picks1 = current[0]
            cnts2, vals2, picks2 = current[1]
            assert cnts1 == cnts2
            dataset.append((cnts1, vals1, picks1, vals2, picks2))
            current = []
    return dataset


def compute_score(vals, picks):
    """Compute the score of the selection."""
    assert len(vals) == len(picks)
    return np.sum([v * p for v, p in zip(vals, picks)])


def gen_choices(cnts, idx=0, choice=[]):
    """Generate all the valid choices.
    It generates both yours and your opponent choices.
    """
    if idx >= len(cnts):
        return [(choice[:], [n - c for n, c in zip(cnts, choice)]),]
    choices = []
    for c in range(cnts[idx] + 1):
        choice.append(c)
        choices += gen_choices(cnts, idx + 1, choice)
        choice.pop()
    return choices


def main():
    parser = argparse.ArgumentParser(
        description='A script to compute Pareto efficiency')
    parser.add_argument('--log_file', type=str, default='',
        help='location of the log file')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')

    args = parser.parse_args()
    domain = get_domain(args.domain)

    dataset = parse_log(args.log_file, domain)

    avg_agree, avg_can_improve = 0, 0
    avg_score1, avg_score2 = 0, 0
    avg_max_score1, avg_max_score2 = 0, 0
    for cnts, vals1, picks1, vals2, picks2 in dataset:
        if np.min(picks1) == -1 or np.min(picks2) == -1:
            continue
        agree = True
        for p1, p2, n in zip(picks1, picks2, cnts):
            agree = agree and (p1 + p2 == n)
        if not agree:
            continue

        avg_agree += 1
        score1 = compute_score(vals1, picks1)
        score2 = compute_score(vals2, picks2)
        choices = gen_choices(cnts)
        can_improve = False
        for cand1, cand2 in choices:
            cand_score1 = compute_score(vals1, cand1)
            cand_score2 = compute_score(vals2, cand2)
            if (cand_score1 > score1 and cand_score2 >= score2) or (cand_score1 >= score1 and cand_score2 > score2):
                can_improve = True

        avg_score1 += score1
        avg_score2 += score2
        avg_can_improve += int(can_improve)

    print('pareto opt (%%)\t:\t%.2f' %  (100. * (1 - avg_can_improve / avg_agree)))
    print('agree (%%)\t:\t%.2f' % (100. * avg_agree / len(dataset)))
    print('score (all)\t:\t%.2f vs. %.2f' % (
        1. * avg_score1 / len(dataset), 1. * avg_score2 / len(dataset)))
    print('score (agreed)\t:\t%.2f vs. %.2f' % (
        1. * avg_score1 / avg_agree, 1. * avg_score2 / avg_agree))


if __name__ == '__main__':
    main()
