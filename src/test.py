# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Performs evaluation of the model on the test dataset.
"""

import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data
import utils
from engine import Engine, Criterion


def main():
    parser = argparse.ArgumentParser(description='testing script')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--model_file', type=str,
        help='pretrained model file')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--hierarchical', action='store_true', default=False,
        help='use hierarchical model')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    args = parser.parse_args()

    device_id = utils.use_cuda(args.cuda)
    utils.set_seed(args.seed)

    corpus = data.WordCorpus(args.data, freq_cutoff=args.unk_threshold, verbose=True)
    model = utils.load_model(args.model_file)

    crit = Criterion(model.word_dict, device_id=device_id)
    sel_crit = Criterion(model.item_dict, device_id=device_id,
        bad_toks=['<disconnect>', '<disagree>'])


    testset, testset_stats = corpus.test_dataset(args.bsz, device_id=device_id)
    test_loss, test_select_loss = 0, 0

    N = len(corpus.word_dict)
    for batch in testset:
        # run forward on the batch, produces output, hidden, target,
        # selection output and selection target
        out, hid, tgt, sel_out, sel_tgt = Engine.forward(model, batch, volatile=False)

        # compute LM and selection losses
        test_loss += tgt.size(0) * crit(out.view(-1, N), tgt).data[0]
        test_select_loss += sel_crit(sel_out, sel_tgt).data[0]

    test_loss /= testset_stats['nonpadn']
    test_select_loss /= len(testset)
    print('testloss %.3f | testppl %.3f' % (test_loss, np.exp(test_loss)))
    print('testselectloss %.3f | testselectppl %.3f' % (test_select_loss, np.exp(test_select_loss)))


if __name__ == '__main__':
    main()
