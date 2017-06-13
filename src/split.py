# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import argparse
import data
import re


item_pattern = re.compile('^item([0-9])=([0-9\-])+$')


def find(tokens, tag):
    for i, t in enumerate(tokens):
        if t == tag:
            return i
    assert False


def invert(cnts, sel):
    inv_sel = []
    for s in sel:
        match = item_pattern.match(s)
        i = int(match.groups()[0])
        v = int(match.groups()[1])
        inv_sel.append('item%d=%d' % (i, cnts[i] - v))
    return inv_sel


def dialog_len(line):
    tokens = line.split(' ')
    n = len([t for t in tokens if t in ('YOU:', 'THEM:')])
    return 'dialoglen: %d' % n


def select(line):
    tokens = line.split(' ')
    n = 0
    for i in range(1, len(tokens)):
        if tokens[i] == '<selection>' and tokens[i - 1] == 'YOU:':
            n = 1
            break
    return 'botselect: %d' % n


def conv(line):
    tokens = line.split(' ')
    context = tokens[3:9]
    assert tokens[9] in ('YOU:', 'THEM:')
    inv = tokens[9] == 'THEM:'
    sel_start = find(tokens, '<selection>') + 1
    cnts = [int(n) for n in context[0::2]]
    if tokens[sel_start] == 'no' or tokens[sel_start] == '<no_agreement>':
        selection = ['<no_agreement>'] * 3
    else:
        selection = tokens[sel_start: sel_start + 3]
        if inv:
            selection = invert(cnts, selection)
    return 'debug: %s %s %s' % (' '.join(context), ' '.join(selection), ' '.join(selection))


def main():
    parser = argparse.ArgumentParser(
        description='A script to compute Pareto efficiency')
    parser.add_argument('--log_file', type=str, default='',
        help='location of the log file')
    parser.add_argument('--output_file', type=str, default='',
        help='location of the log file')
    parser.add_argument('--bot_name', type=str, default='',
        help='bot name')
    args = parser.parse_args()

    lines = data.read_lines(args.log_file)
    bots = dict()
    for line in lines:
        if line.startswith(args.bot_name + '1') or \
            line.startswith(args.bot_name + '2'):
            bots[line.split(' ')[2]] = line

    humans = dict()
    for line in lines:
        if line.startswith('human'):
            i = line.split(' ')[2]
            if i in bots:
                humans[i] = line

    with open(args.output_file, 'w') as f:
        for i in bots.keys():
            if i in humans:
                print(dialog_len(bots[i]), file=f)
                print(select(bots[i]), file=f)
                print(conv(bots[i]), file=f)
                print(conv(humans[i]), file=f)
                print('-' * 80, file=f)

if __name__ == '__main__':
    main()
