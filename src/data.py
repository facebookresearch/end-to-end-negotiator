# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import sys
import pdb
import copy
import re
from collections import OrderedDict

import torch
import numpy as np


SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

STOP_TOKENS = [
    '<eos>',
    '<selection>',
]


def get_tag(tokens, tag):
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def read_lines(file_name):
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        self.idx2word = []
        if init:
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def i2w(self, idx):
        return [self.idx2word[i] for i in idx]

    def w2i(self, words):
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word):
        unk = self.word2idx.get('<unk>', None)
        return self.word2idx.get(word, unk)

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def read_tag(file_name, tag, freq_cutoff=-1, init_dict=True):
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = Dictionary(init=init_dict)
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff:
                dictionary.add_word(token)
        return dictionary

    def from_file(file_name, freq_cutoff):
        assert os.path.exists(file_name)
        word_dict = Dictionary.read_tag(file_name, 'dialogue', freq_cutoff=freq_cutoff)
        item_dict = Dictionary.read_tag(file_name, 'output', init_dict=False)
        context_dict = Dictionary.read_tag(file_name, 'input', init_dict=False)
        return word_dict, item_dict, context_dict


class WordCorpus(object):
    def __init__(self, path, freq_cutoff=2, train='train.txt',
        valid='val.txt', test='test.txt', verbose=False):
        self.verbose = verbose
        # Only add words from the train dataset
        self.word_dict, self.item_dict, self.context_dict = Dictionary.from_file(
            os.path.join(path, train),
            freq_cutoff=freq_cutoff)

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        self.output_length = max([len(x[2]) for x in self.train])

    def tokenize(self, file_name):
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_idxs = self.context_dict.w2i(get_tag(tokens, 'input'))
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            item_idxs = self.item_dict.w2i(get_tag(tokens, 'output'))
            dataset.append((input_idxs, word_idxs, item_idxs))
            total += len(input_idxs) + len(word_idxs) + len(item_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.train), bsz,
            shuffle=shuffle, device_id=device_id)

    def valid_dataset(self, bsz, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.valid), bsz,
            shuffle=shuffle, device_id=device_id)

    def test_dataset(self, bsz, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle,
            device_id=device_id)

    def _split_into_batches(self, dataset, bsz, shuffle=True, device_id=None):
        if shuffle:
            random.shuffle(dataset)

        # Sort and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, items = [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                items.append(dataset[j][2])

            max_len = len(words[-1])

            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                words[j] += [pad] * (max_len - len(words[j]))

            ctx = torch.LongTensor(inputs).transpose(0, 1).contiguous()
            data = torch.LongTensor(words).transpose(0, 1).contiguous()
            sel_tgt = torch.LongTensor(items).transpose(0, 1).contiguous().view(-1)
            if device_id is not None:
                ctx = ctx.cuda(device_id)
                data = data.cuda(device_id)
                sel_tgt = sel_tgt.cuda(device_id)

            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats
