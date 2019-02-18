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

    def read_tag(domain, file_name, tag, freq_cutoff=-1, init_dict=True):
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


class ItemDictionary(Dictionary):
    def __init__(self, selection_size, init=True):
        super(ItemDictionary, self).__init__(init)
        self.selection_size = selection_size

    def w2i(self, words, inv=False):
        # pick last selection_size if inv=True, otherwise first selection_size
        words = words[self.selection_size:] if inv else words[:self.selection_size]
        token = ' '.join(words)
        return self.word2idx[token]

    def read_tag(domain, file_name, tag, init_dict=False):
        dictionary = ItemDictionary(domain.selection_length() // 2, init=False)

        def generate(item_id, selection=[]):
            if item_id >= dictionary.selection_size:
                dictionary.add_word(' '.join(selection))
                return
            for i in range(5):
                selection.append('item%d=%d' % (item_id, i))
                generate(item_id + 1, selection)
                selection.pop()

        generate(0)

        for token in ['<disagree>', '<no_agreement>', '<disconnect>']:
            dictionary.add_word(' '.join([token] * dictionary.selection_size))

        return dictionary


class CountDictionary(Dictionary):
    def __init__(self, init=True):
        super(CountDictionary, self).__init__(init)

    def get_key(words):
        key = '_'.join(words[i] for i in range(0, len(words), 2))
        return key

    def get_idx(self, words):
        key = CountDictionary.get_key(words)
        return self.word2idx[key]

    def read_tag(domain, file_name, tag, init_dict=False):
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                key = CountDictionary.get_key(tokens)
                token_freqs[key] = token_freqs.get(key, 0) + 1
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
        dictionary = CountDictionary(init=init_dict)
        for token, freq in token_freqs:
                dictionary.add_word(token)
        return dictionary


def create_dicts_from_file(domain, file_name, freq_cutoff):
    assert os.path.exists(file_name)
    word_dict = Dictionary.read_tag(domain, file_name, 'dialogue', freq_cutoff=freq_cutoff)
    item_dict = ItemDictionary.read_tag(domain, file_name, 'output', init_dict=False)
    item_dict_old = Dictionary.read_tag(domain, file_name, 'output', init_dict=False)
    context_dict = Dictionary.read_tag(domain, file_name, 'input', init_dict=False)
    count_dict = CountDictionary.read_tag(domain, file_name, 'input', init_dict=False)
    return word_dict, item_dict, context_dict, item_dict_old, count_dict


class WordCorpus(object):
    def __init__(self, domain, path, freq_cutoff=2, train='train.txt',
        valid='val.txt', test='test.txt', verbose=False, sep_sel=False):
        self.domain = domain
        self.verbose = verbose
        self.sep_sel = sep_sel
        # Only add words from the train dataset
        self.word_dict, self.item_dict, self.context_dict, self.item_dict_old, self.count_dict = create_dicts_from_file(
            domain,
            os.path.join(path, train),
            freq_cutoff=freq_cutoff)

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[2]) for x in self.train])

    def tokenize(self, file_name):
        lines = read_lines(file_name)
        random.shuffle(lines)

        def make_mask(choices, inv=False):
            items = torch.Tensor([self.item_dict.w2i(c, inv=inv) for c in choices]).long()
            mask = torch.Tensor(len(self.item_dict)).zero_()
            mask.scatter_(0, items, torch.Tensor(items.size(0)).fill_(1))
            return mask.unsqueeze(0)

        def make_indexes(choices):
            items = torch.Tensor([self.item_dict.w2i(c) for c in choices]).long()
            return items


        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_idxs = self.context_dict.w2i(get_tag(tokens, 'input'))
            count_idx = self.count_dict.get_idx(get_tag(tokens, 'input'))
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            item_idx = self.item_dict.w2i(get_tag(tokens, 'output'), inv=False)
            item_idx_inv = self.item_dict.w2i(get_tag(tokens, 'output'), inv=True)
            items = self.item_dict_old.w2i(get_tag(tokens, 'output'))

            #valid_choices = self.domain.generate_choices(get_tag(tokens, 'input'), with_disagreement=False)
            #valid_mask = make_mask(valid_choices)
            partner_input_idxs = self.context_dict.w2i(get_tag(tokens, 'partner_input'))
            if self.sep_sel:
                dataset.append((input_idxs, word_idxs, items, partner_input_idxs, count_idx))
            else:
                dataset.append((input_idxs, word_idxs, [item_idx, item_idx_inv],
                    partner_input_idxs, count_idx))

            total += len(input_idxs) + len(word_idxs) + len(partner_input_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle)

    def valid_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle)

    def test_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle)

    def _split_into_batches(self, dataset, bsz, shuffle=True):
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
            inputs, words = [], []
            if self.sep_sel:
                items = []
            else:
                items = [[], []]
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                if self.sep_sel:
                    items.append(dataset[j][2])
                else:
                    for k in range(2):
                        items[k].append(dataset[j][2][k])

            max_len = len(words[-1])

            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                words[j] += [pad] * (max_len - len(words[j]))

            ctx = torch.Tensor(inputs).long().transpose(0, 1).contiguous()
            data = torch.Tensor(words).long().transpose(0, 1).contiguous()
            if self.sep_sel:
                sel_tgt = torch.Tensor(items).long().transpose(0, 1).contiguous().view(-1)
            else:
                sel_tgt = [torch.Tensor(it).long().view(-1) for it in items]

            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats


class SentenceCorpus(WordCorpus):
    def _split_into_sentences(self, dataset):
        stops = [self.word_dict.get_idx(w) for w in ['YOU:', 'THEM:']]
        sent_dataset = []
        for ctx, words, items, partner_ctx, count_idx in dataset:
            sents, current = [], []
            for w in words:
                if w in stops:
                    if len(current) > 0:
                        sents.append(current)
                    current = []
                current.append(w)
            if len(current) > 0:
                sents.append(current)
            sent_dataset.append((ctx, sents, items, partner_ctx, count_idx))
        # Sort by numper of sentences in a dialog
        sent_dataset.sort(key=lambda x: len(x[1]))

        return sent_dataset

    def _make_reverse_idxs(self, inpts, lens):
        idxs = []
        for inpt, ln in zip(inpts, lens):
            idx = torch.Tensor(inpt.size(0), inpt.size(1), 1).long().fill_(-1)
            for i in range(inpt.size(1)):
                arngmt = torch.Tensor(inpt.size(0), 1, 1).long()
                for j in range(arngmt.size(0)):
                    arngmt[j][0][0] = j if j > ln[i] else ln[i] - j
                idx.narrow(1, i, 1).copy_(arngmt)
            idxs.append(idx)
        return idxs

    def _make_hidden_idxs(self, lens):
        idxs = []
        for s, ln in enumerate(lens):
            idx = torch.Tensor(1, ln.size(0), 1).long()
            for i in range(ln.size(0)):
                idx[0][i][0] = ln[i]
            idxs.append(idx)
        return idxs

    def _split_into_batches(self, dataset, bsz, shuffle=True):
        if shuffle:
            random.shuffle(dataset)

        dataset = self._split_into_sentences(dataset)

        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        i = 0
        while i < len(dataset):
            dial_len = len(dataset[i][1])

            ctxs, dials, partner_ctxs, count_idxs = [], [], [], []
            if self.sep_sel:
                items = []
            else:
                items = [[], []]
            for _ in range(bsz):
                if i >= len(dataset) or len(dataset[i][1]) != dial_len:
                    break
                ctxs.append(dataset[i][0])
                dials.append(dataset[i][1])
                if self.sep_sel:
                    items.append(dataset[i][2])
                else:
                    for j in range(2):
                        items[j].append(dataset[i][2][j])
                partner_ctxs.append(dataset[i][3])
                count_idxs.append(dataset[i][4])
                i += 1

            inpts, lens, tgts = [], [], []
            for s in range(dial_len):
                batch = []
                for dial in dials:
                    batch.append(dial[s])
                if s + 1 < dial_len:
                    # add YOU:/THEM: as the last tokens in order to connect sentences
                    for j in range(len(batch)):
                        batch[j].append(dials[j][s + 1][0])
                else:
                    # add <pad> after <selection>
                    for j in range(len(batch)):
                        batch[j].append(pad)


                max_len = max([len(sent) for sent in batch])
                ln = torch.LongTensor(len(batch))
                for j in range(len(batch)):
                    stats['n'] += max_len
                    stats['nonpadn'] += len(batch[j]) - 1
                    ln[j] = len(batch[j]) - 2
                    batch[j] += [pad] * (max_len - len(batch[j]))
                sent = torch.Tensor(batch).long().transpose(0, 1).contiguous()
                inpt = sent.narrow(0, 0, sent.size(0) - 1)
                tgt = sent.narrow(0, 1, sent.size(0) - 1).view(-1)
                inpts.append(inpt)
                lens.append(ln)
                tgts.append(tgt)

            ctx = torch.Tensor(ctxs).long().transpose(0, 1).contiguous()
            partner_ctx = torch.Tensor(partner_ctxs).long()#.view(-1).contiguous() #.transpose(0, 1).contiguous()
            if self.sep_sel:
                sel_tgt = torch.Tensor(items).long().view(-1)
            else:
                sel_tgt = [torch.Tensor(it).long().view(-1) for it in items]

            cnt = torch.Tensor(count_idxs).long()

            rev_idxs = self._make_reverse_idxs(inpts, lens)
            hid_idxs = self._make_hidden_idxs(lens)

            batches.append((ctx, partner_ctx, inpts, lens, tgts, sel_tgt, rev_idxs, hid_idxs, cnt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats


class PhraseCorpus(WordCorpus):
    def tokenize(self, file_name):
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            dialog, items = get_tag(tokens, 'dialogue'), get_tag(tokens, 'output')

            for words in re.split(r'(?:YOU|THEM):', ' '.join(dialog))[1:-1]:
                if words is '':
                    continue
                words = words.strip().split(' ')
                word_idxs = self.word_dict.w2i(words)
                item_idxs = self.item_dict.w2i(items)
                dataset.append(word_idxs)
                total += len(word_idxs) + len(item_idxs)
                unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def _split_into_batches(self, dataset, bsz, shuffle=True):
        if shuffle:
            random.shuffle(dataset)

        # Sort and pad
        dataset.sort(key=lambda x: len(x))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        for i in range(0, len(dataset), bsz):
            words = []
            for j in range(i, min(i + bsz, len(dataset))):
                words.append(dataset[j])

            max_len = len(words[-1])

            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                words[j] += [pad] * (max_len - len(words[j]))

            data = torch.Tensor(words).transpose(0, 1).long().contiguous()
            # inpt = data.narrow(0, CONTEXT_LENGTH, data.size(0) - 1 - CONTEXT_LENGTH)

            batches.append(data)

        if shuffle:
            random.shuffle(batches)

        return batches, stats
