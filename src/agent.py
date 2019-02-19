# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections import defaultdict

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from dialog import DialogLogger
import vis
import domain
from engines import Criterion
import math
from collections import Counter


class Agent(object):
    """ Agent's interface. """
    def feed_context(self, ctx):
        pass

    def read(self, inpt):
        pass

    def write(self):
        pass

    def choose(self):
        pass

    def update(self, agree, reward, partner_choice):
        pass


class RnnAgent(Agent):
    def __init__(self, model, args, name='Alice', allow_no_agreement=True, train=False, diverse=False):
        super(RnnAgent, self).__init__()
        self.model = model
        self.model.eval()
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)
        self.allow_no_agreement = allow_no_agreement

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()


    def _encode(self, inpt, dictionary):
        encoded = torch.Tensor(dictionary.w2i(inpt)).long().unsqueeze(1)
        return encoded

    def _decode(self, out, dictionary):
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context):
        self.lang_hs = []
        self.sents = []
        self.words = []
        self.context = context
        self.ctx = self._encode(context, self.model.context_dict)
        self.ctx_h = self.model.forward_context(Variable(self.ctx))
        self.lang_h = self.model.zero_h(1, self.model.args.nhid_lang)

    def feed_partner_context(self, partner_context):
        pass

    def update(self, agree, reward, choice=None, partner_choice=None,
            partner_input=None, max_partner_reward=None):
        pass

    def read(self, inpt):
        self.sents.append(Variable(self._encode(['THEM:'] + inpt, self.model.word_dict)))
        inpt = self._encode(inpt, self.model.word_dict)
        lang_hs, self.lang_h = self.model.read(Variable(inpt), self.lang_h, self.ctx_h)
        self.lang_hs.append(lang_hs.squeeze(1))
        self.words.append(self.model.word2var('THEM:').unsqueeze(0))
        self.words.append(Variable(inpt))
        assert (torch.cat(self.words).size(0) == torch.cat(self.lang_hs).size(0))

    def write(self, max_words=100):
        acts, outs, self.lang_h, lang_hs = self.model.write(self.lang_h, self.ctx_h,
                                                            max_words, self.args.temperature)
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:').unsqueeze(0))
        self.words.append(outs)
        self.sents.append(torch.cat([self.model.word2var('YOU:').unsqueeze(1), outs], 0))
        assert (torch.cat(self.words).size(0) == torch.cat(self.lang_hs).size(0))
        return self._decode(outs, self.model.word_dict)

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose(self, sample=False):
        sents = self.sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, Variable(self.ctx))

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].item())
        prob = F.softmax(choice_logit, dim=0)

        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=0).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.item()]

        # Pick only your choice
        return choices[idx.item()][:self.domain.selection_length()], logprob, p_agree.item()

    def choose(self):
        choice, _, _ = self._choose()
        return choice


class HierarchicalAgent(RnnAgent):
    def __init__(self, model, args, name='Alice'):
        super(HierarchicalAgent, self).__init__(model, args, name)
        self.vis = False

    def feed_context(self, context):
        self.lang_hs = []
        self.sents = []
        self.context = context
        self.ctx = self._encode(context, dictionary=self.model.context_dict)
        self.ctx_h = self.model.forward_context(Variable(self.ctx)).squeeze(0)
        self.strat_h = self.model.zero_h(1, self.model.args.nhid_strat).squeeze(0)
        self.lang_h = self.model.zero_h(1, self.model.args.nhid_lang).squeeze(0)
        self.lang_attn = self.model.zero_h(1, self.model.args.nhid_lang).squeeze(0)

    def read(self, inpt):
        inpt = ['THEM:'] + inpt
        inpt = Variable(self._encode(inpt, self.model.word_dict))
        self.sents.append(inpt)
        self.strat_h, self.lang_h, lang_hs, self.lang_attn, attn_p = self.model.read(
            inpt, self.strat_h, self.lang_h, self.ctx_h, self.lang_attn)
        self.lang_hs.append(lang_hs)
        if self.vis:
            vis.plot_attn(self._decode(inpt, self.model.word_dict), attn_p)

    def write(self):
        _, outs, self.strat_h, self.lang_h, lang_hs, self.lang_attn, attn_p = self.model.write(
            self.strat_h, self.lang_h, self.ctx_h, self.lang_attn, 100, self.args.temperature)
        self.lang_hs.append(lang_hs)
        self.sents.append(outs)
        # remove 'YOU:'
        if self.vis:
            vis.plot_attn(self._decode(outs, self.model.word_dict), attn_p)
        outs = outs.narrow(0, 1, outs.size(0) - 1)

        return self._decode(outs, self.model.word_dict)

    def _make_idxs(self):
        lens, hid_idxs, rev_idxs = [], [], []
        for sent in self.sents:
            assert sent.size(1) == 1
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
        return lens, hid_idxs, rev_idxs

    def _make_choice_logits(self):
        if len(self.sents) == 0:
            return Variable(torch.Tensor(1, self.model.output_length * len(self.model.item_dict)).fill_(0))

        inpt_embs = self.model.forward_embedding(self.sents)
        lens, hid_idxs, rev_idxs = self._make_idxs()
        logits, attn_p = self.model.forward_selection(
            inpt_embs, self.lang_hs, lens,
            self.strat_h.unsqueeze(1), self.ctx_h.unsqueeze(1),
            rev_idxs, hid_idxs)

        if self.vis:
            sent_p, word_ps = attn_p
            for sent, word_p in zip(self.sents, word_ps):
                vis.plot_attn(self._decode(sent, self.model.word_dict), word_p)
            vis.plot_attn(['%d' % i for i in range(len(word_ps))], sent_p)

        return logits

    def _choose(self, sample=False, logits=None):
        if not logits:
            logits = self._make_choice_logits()

        logits = logits.view(self.domain.selection_length(), -1)

        choices = self.domain.generate_choices(self.context)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(logits[i].squeeze(0), 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(1, keepdim=True).data[0])
        prob = F.softmax(choice_logit)
        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]].data[0]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree


class RnnRolloutAgent(RnnAgent):
    def __init__(self, model, args, name='Alice', train=False):
        super(RnnRolloutAgent, self).__init__(model, args, name)
        self.ncandidate = 5
        self.nrollout = 3
        self.rollout_len = 100

    def write(self, max_words):
        best_score = -1
        res = None

        for _ in range(self.ncandidate):
            _, move, move_lang_h, move_lang_hs = self.model.write(
                self.lang_h, self.ctx_h, 100, self.args.temperature)

            is_selection = len(move) == 1 and \
                self.model.word_dict.get_word(move.data[0][0]) == '<selection>'

            score = 0
            for _ in range(self.nrollout):
                combined_lang_hs = self.lang_hs + [move_lang_hs]
                combined_words = self.words + [self.model.word2var('YOU:'), move]

                if not is_selection:
                    # Complete the conversation with rollout_length samples
                    _, rollout, _, rollout_lang_hs = self.model.write(
                        move_lang_h, self.ctx_h, self.rollout_len, self.args.temperature,
                        stop_tokens=['<selection>'], resume=True)
                    combined_lang_hs += [rollout_lang_hs]
                    combined_words += [rollout]

                # Choose items
                rollout_score = None

                combined_lang_hs = torch.cat(combined_lang_hs)
                combined_words = torch.cat(combined_words)
                rollout_choice, _, p_agree = self._choose(combined_lang_hs, combined_words, sample=False)
                rollout_score = self.domain.score(self.context, rollout_choice)
                score += p_agree * rollout_score

            # Take the candidate with the max expected reward
            if score > best_score:
                res = (move, move_lang_h, move_lang_hs)
                best_score = score

        outs, lang_h, lang_hs = res
        self.lang_h = lang_h
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:'))
        self.words.append(outs)
        return self._decode(outs, self.model.word_dict)


class StrategyAgent(HierarchicalAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=False):
        super(StrategyAgent, self).__init__(model, args, name)

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

    def feed_context(self, context):
        self.lang_hs = []
        self.sents = []
        self.sents2 = []
        self.context = context
        self.ctx = self._encode(context, dictionary=self.model.context_dict)
        self.ctx_h = self.model.forward_context(Variable(self.ctx))
        self.sent_attn_h = self.model.zero_h(1, 2 * self.model.args.nhid_attn)
        self.lang_h = self.model.zero_h(1, self.model.args.nhid_lang)
        self.strat_h = self.model.zero_h(1, self.model.args.nhid_strat)
        self.strat_hs = []
        #self.sel_h = self.model.zero_h(1, self.model.selection_size)
        #self.future_h, self.next_strat_h = self.model.future_strat(self.strat_h, self.model.future_strat.zero_h(1))

    def feed_partner_context(self, partner_context):
        self.partner_context = partner_context
        self_partner_ctx = self._encode(partner_context, dictionary=self.model.context_dict)

    def _make_idxs2(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose2(self, sample=False):
        sents = self.sents2[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs2(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, Variable(self.ctx))

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree.data[0]

    def choose(self):
        choice, _, _ = self._choose2()
        return choice

    def update_attention(self):
        # recompute lang_attn_h
        if self.sents[-1][0].data[0] == self.model.word_dict.word2idx['<selection>']:
            return
        lens, rev_idxs, hid_idxs = self._make_idxs(self.sents, self.lang_hs)
        (self.sent_attn_h, sent_attn_p), (word_attn_hs, word_attn_ps) = self.model.attn(
            self.strat_h, self.lang_hs, lens, rev_idxs, hid_idxs)

        if self.vis:
            sents = [self._decode(sent, self.model.word_dict) for sent in self.sents]
            #vis.plot_attn(sents, sent_attn_p, word_attn_ps)
        #import pdb; pdb.set_trace()

    def update_strategy(self):
        # update current strategy
        self.strat_h = self.model.strategy(self.strat_h, self.sent_attn_h, self.lang_h, self.ctx_h)
        self.strat_hs.append(self.strat_h)

        # update future strategy
        #self.next_strat_h, self.future_h = self.model.future_strat(self.strat_h, self.future_h)

        #self.plot_strategy()
        #self.plot_partner_context()
        #import pdb; pdb.set_trace()

    def plot_strategy(self):
        choices = self.domain.generate_choices(self.context)

        def get_prob(logits, attn_p):
            choices_logits = []
            for i in range(self.domain.selection_length()):
                idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
                idxs = Variable(torch.Tensor(idxs).long())
                choices_logits.append(torch.gather(logits[i].squeeze(0), 0, idxs).unsqueeze(1))

            if vis:
                p = F.softmax(logits)
                labels = ['%s' % s for s in self.model.item_dict.idx2word]
                #for i in range(3):
                #    vis.plot_distribution(p[i], labels, 'Item%d' % i)


            choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
            choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
            return F.softmax(choice_logit), attn_p


        actual_prob, attn_p = get_prob(*self.model.forward_selection(self.strat_hs, self.ctx_h))
        next_actual_prob, next_attn_p = get_prob(*self.model.forward_selection(self.strat_hs + [self.next_strat_h], self.ctx_h))
        #predicted_prob = get_prob(self.sel_h.view(self.domain.selection_length(), -1))

        labels = []
        for choice in choices:
            if choice[0] == '<no_agreement>':
                labels.append('no_agree')
            elif choice[0] == '<disconnect>':
                labels.append('discon')
            else:
                you = [str(self.domain.parse_choice(c)[1]) for c in choice[:3]]
                them = [str(self.domain.parse_choice(c)[1]) for c in choice[3:]]
                labels.append('(%s)' % (','.join(you)))
        title = 'Your input %s' % (' '.join(['%s=(%s:%s)' % (DialogLogger.CODE2ITEM[i][1],
            self.context[2 * i + 0], self.context[2 * i + 1]) \
            for i in range(len(self.context) // 2)]))

        if self.vis:
            vis.plot_distribution(actual_prob, labels, 'Actual: ' + title)
            vis.plot_distribution(next_actual_prob, labels, 'Next Actual: ' + title)
            if attn_p.size(1) > 1:
                labels = ['%d' % i for i in range(attn_p.size(1))]
                #vis.plot_distribution(attn_p.squeeze(0), labels, 'Strategy Attention')
            #vis.plot_distribution(predicted_prob, labels, 'Predicted: ' + title)

            #score_out = F.softmax(self.model.forward_score(self.strat_hs, self.ctx_h))
            #labels = ['_%d' % i for i in range(11)]
            #vis.plot_distribution(score_out.squeeze(0), labels, 'Score')

    def plot_partner_context(self):
        logits = self.model.forward_partner_context(self.strat_hs)
        prob = F.softmax(logits)

        n = len(self.partner_context) // 2

        title = 'Partner input: %s' % (' '.join(['%s=(%s:%s)' % (
            DialogLogger.CODE2ITEM[i][1],
            self.partner_context[2 * i + 0],
            self.partner_context[2 * i + 1]) for i in range(n)]))
        legend = [DialogLogger.CODE2ITEM[i][1] for i in range(n)]
        rownames = self.model.context_dict.idx2word
        prob = prob.transpose(0, 1).contiguous()
        if self.vis:
            vis.plot_context(prob, rownames, title, legend)

    def read(self, inpt):
        # update strategy
        self.update_strategy()
        if self.vis:
            import pdb; pdb.set_trace()

        self.sents.append(Variable(self._encode(inpt, self.model.word_dict)))
        # prepend 'THEM:'
        inpt = ['THEM:'] + inpt
        inpt = Variable(self._encode(inpt, self.model.word_dict))
        self.sents2.append(inpt)

        # read up the sentence
        lang_hs, self.lang_h = self.model.read(inpt, self.strat_h)

        # store hidden states and the sentence
        self.lang_hs.append(lang_hs)

        # update attention
        self.update_attention()

    def write(self, max_words=100):
        self.update_strategy()
        if self.vis:
            import pdb; pdb.set_trace()

        # generate new sentence
        _, outs, lang_hs, self.lang_h, topks = self.model.write(
            self.strat_h, max_words, self.args.temperature)

        if self.vis:
            if outs.size(0) > 1:
                starts = ['YOU:'] + self.model.word_dict.i2w(outs.narrow(0, 0, outs.size(0) - 1).data[:, 0])
                for i in range(len(starts)):
                    start = starts[i]
                    prob, words = topks[i]
                    words = self.model.word_dict.i2w(words.data)
                    words = ['%s_' % w for w in words]
                    vis.plot_distribution(prob, words, start)

        self.sents2.append(torch.cat([self.model.word2var('YOU:').unsqueeze(1), outs], 0))
        self.sents.append(outs)
        outs = self._decode(outs, self.model.word_dict)

        #inpt = ['YOU:'] + outs
        #inpt = Variable(self._encode(outs, self.model.word_dict))
        #lang_hs, self.lang_h = self.model.read(inpt, self.strat_h)
        self.lang_hs.append(lang_hs)
        self.update_attention()

        return outs

        # store lang_hs and outs
        self.lang_hs.append(lang_hs)
        self.sents.append(outs)

        self.update_attention()

        # if self.vis:
        #    vis.plot_attn(self._decode(outs, self.model.word_dict), attn_p)

        # decode to English
        return self._decode(outs, self.model.word_dict)

    def _make_idxs(self, sents, lang_hs):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent, lang_hs in zip(sents, lang_hs):
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0)).long()
            lens.append(ln)
            idx = torch.Tensor(lang_hs.size(0) , 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(None)
        return lens, rev_idxs, hid_idxs

    def _choose(self, strat_hs, sample=False):
        logits, _ = self.model.forward_selection(strat_hs, self.ctx_h)

        logits = logits.view(self.domain.selection_length(), -1)

        choices = self.domain.generate_choices(self.context)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(logits[i].squeeze(0), 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)
        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]].data[0]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree



class StrategyRolloutAgent(StrategyAgent):
    def __init__(self, models, args, name='Alice'):
        model, forward_model = models
        super(StrategyRolloutAgent, self).__init__(model, args, name)
        self.ncandidate = 10
        self.nrollout = 10
        self.rollout_len = 5
        self.forward_model = forward_model
        self.forward_model.eval()

    def feed_context(self, context):
        super(StrategyRolloutAgent, self).feed_context(context)
        self.forward_h = self.forward_model.zero_h(1, self.forward_model.hidden_size).squeeze(0)

    def update_strategy(self):
        super(StrategyRolloutAgent, self).update_strategy()
        forward_inpt = torch.cat([self.strat_h, self.ctx_h], 1)
        self.forward_h = self.forward_model.step(forward_inpt, self.forward_h)

    def write(self):
        self.update_strategy()

        best_score = -1
        res = None

        for _ in range(self.ncandidate):
            _, move, move_lang_hs, move_lang_h = self.model.write(
                self.strat_h, 100, self.args.temperature)

            is_selection = len(move) == 1 and \
                self.model.word_dict.get_word(move.data[0][0]) == '<selection>'

            all_lang_hs = self.lang_hs + [move_lang_hs]
            all_sents = self.sents + [move]
            lens, rev_idxs, hid_idxs = self._make_idxs(all_sents, all_lang_hs)
            (attn_h, _), _ = self.model.attn(self.strat_h, all_lang_hs, lens, rev_idxs, hid_idxs)
            move_strat_h = self.model.strategy(self.strat_h, attn_h, move_lang_h, self.ctx_h)

            forward_h = self.forward_h.expand(self.nrollout, self.forward_h.size(1))
            strat_h = move_strat_h.expand(self.nrollout, self.move_strat_h.size(1))
            ctx_h = self.ctx_h.expand(self.nrollout, self.ctx_h.size(1))

            for i in range(self.rollout_len):
                forward_inpt = torch.cat([strat_h, ctx_h], 1)
                forward_h = self.forward_model.step(forward_inpt, forward_h)
                strat_h = forward_h

            total_score = 0
            for _ in range(self.nrollout):
                forward_h = self.forward_h
                strat_h = move_strat_h
                gamma = 1
                for i in range(self.rollout_len):
                    forward_inpt = torch.cat([strat_h, self.ctx_h], 1)
                    forward_h = self.forward_model.step(forward_inpt, forward_h)
                    strat_h = forward_h

                    choice, _, agree_prob = self._choose([strat_h])
                    score = self.domain.score(self.context, choice)
                    total_score += gamma * agree_prob * score

                    gamma *= 0.99

            if total_score > best_score:
                best_score = total_score
                res = (move, move_lang_hs, move_lang_h)

        out, lang_hs, self.lang_h = res
        self.lang_hs.append(lang_hs)
        self.sents.append(out)

        self.update_attention()

        return self._decode(out, self.model.word_dict)


class RnnRolloutAgent2(RnnAgent):
    def __init__(self, model, args, name='Alice', force_unique_candidates=False, sample_selection=False,
                 allow_no_agreement=True, train=False):
        super(RnnRolloutAgent, self).__init__(model, args, name, allow_no_agreement=allow_no_agreement)
        self.ncandidate = 10 #args.rollout_candidates
        self.nrollout = 16 #args.rollout_bsz
        self.rollout_len = 100
        self.max_attempts = 100
        self.force_unique_candidates = force_unique_candidates
        self.sample_selection = sample_selection

    def update(self, agree, reward, choice=None, partner_choice=None, partner_input=None):
        pass

    def write(self, max_words):
        best_score = -9999
        res = None

        i = 0
        uniq = set()
        while i < self.ncandidate or (best_score < -99990 and i < 100):
            i += 1
            move_string = None
            attempts = 0
            while attempts < self.max_attempts and (not move_string or move_string in uniq or not self.force_unique_candidates):
                _, move, move_lang_h, move_lang_hs = self.model.write(
                    self.lang_h, self.ctx_h, min(max_words, 100), self.args.temperature, first_turn = len(self.words) == 0
                )
                move_string = ' '.join(self._decode(move, self.model.word_dict))
                attempts += 1

            if attempts == self.max_attempts:
                break

            uniq.add(move_string)


            is_selection = len(move) == 1 and \
                self.model.word_dict.get_word(move.data[0][0]) == '<selection>'

            score = 0
            for _ in range(self.nrollout):
                combined_lang_hs = self.lang_hs + [move_lang_hs]
                combined_words = self.words + [self.model.word2var('YOU:'), move]

                if not is_selection and max_words - len(move) > 1:
                    # Complete the conversation with rollout_length samples
                    _, rollout, _, rollout_lang_hs = self.model.write(
                        move_lang_h, self.ctx_h, max_words - len(move), self.args.temperature,
                        stop_tokens=['<selection>'], resume=True)
                    combined_lang_hs += [rollout_lang_hs]
                    combined_words += [rollout]

                # Choose items
                combined_lang_hs = torch.cat(combined_lang_hs)
                combined_words = torch.cat(combined_words)
                rollout_choice, _, p_agree = self._choose(combined_lang_hs, combined_words, sample=self.sample_selection)
                rollout_score = self.domain.score(self.context, rollout_choice)
                if not self.sample_selection:
                    rollout_score = rollout_score * p_agree
                score += rollout_score


            # Take the candidate with the max expected reward
            if score > best_score:
                res = (move, move_lang_h, move_lang_hs)
                best_score = score

        outs, lang_h, lang_hs = res
        self.lang_h = lang_h
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:'))
        self.words.append(outs)
        return self._decode(outs, self.model.word_dict)


class BatchedRolloutAgent(RnnRolloutAgent):
    def __init__(self, model, args, name='Alice'):
        super(BatchedRolloutAgent, self).__init__(model, args, name)
        self.eos = self.model.word_dict.get_idx('<eos>')
        self.eod = self.model.word_dict.get_idx('<selection>')

    def _find(self, seq, tokens):
        n = seq.size(0)
        for i in range(n):
            if seq[i] in tokens:
                return i
        return n

    def write(self):
        batch_outs, batch_lang_hs = self.model.write_batch(
            self.args.rollout_bsz, self.lang_h, self.ctx_h, self.args.temperature)

        counts, scores, states = defaultdict(float), defaultdict(int), defaultdict(list)
        for i in range(self.args.rollout_bsz):
            outs = batch_outs.narrow(1, i, 1).squeeze(1).data.cpu()
            lang_hs = batch_lang_hs.narrow(1, i, 1).squeeze(1)

            # Find the end of the dialogue
            eod_pos = self._find(outs, [self.eod])
            if eod_pos == outs.size(0):
                # Unfinished dialogue, don't count this
                continue

            # Find end of the first utterance
            first_turn_length = self._find(outs, [self.eos, self.eod]) + 1
            move = outs.narrow(0, 0, first_turn_length)
            sent = ' '.join(self.model.word_dict.i2w(move.numpy()))
            sent_lang_hs = lang_hs.narrow(0, 0, first_turn_length + 1)
            lang_h = lang_hs.narrow(0, first_turn_length + 1, 1).unsqueeze(0)

            dialog_lang_hs = lang_hs.narrow(0, 0, eod_pos + 1)
            dialog_words = Variable(outs.narrow(0, 0, eod_pos + 1).cuda())
            choice, _, p_agree = self._choose(
                torch.cat(self.lang_hs + [dialog_lang_hs]),
                torch.cat(self.words + [dialog_words]).squeeze().unsqueeze(1), sample=False)

            # Group by the first utterance
            counts[sent] += 1
            scores[sent] += self.domain.score(self.context, choice) * p_agree
            states[sent] = (lang_h, sent_lang_hs, move)

        for threshold in range(self.args.rollout_count_threshold, -1, -1):
            cands = [k for k in counts if counts[k] >= threshold]
            if cands:
                sent = max(cands, key=lambda k: scores[k] / counts[k])
                lang_h, sent_lang_hs, move = states[sent]
                self.lang_h = lang_h
                self.lang_hs.append(sent_lang_hs)
                self.words.append(self.model.word2var('YOU:'))
                self.words.append(Variable(move.cuda()))
                assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])

                return sent.split(' ')


class RlAgent(RnnAgent):
    def __init__(self, model, args, name='Alice', train=False):
        self.train = train
        super(RlAgent, self).__init__(model, args, name=name)
        self.opt = optim.RMSprop(
            self.model.parameters(),
            lr=args.rl_lr,
            momentum=self.args.momentum)

        self.all_rewards = []

        if self.args.visual:
            self.model_plot = vis.ModulePlot(self.model, plot_weight=False, plot_grad=True)
            self.agree_plot = vis.Plot(['agree',], 'agree', 'agree')
            self.reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'reward', 'reward')
            self.loss_plot = vis.Plot(['loss',], 'loss', 'loss')
            self.agree_reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'agree_reward', 'agree_reward')
        self.t = 0

    def feed_context(self, ctx):
        super(RlAgent, self).feed_context(ctx)
        self.logprobs = []

    def write(self, max_words):
        logprobs, outs, self.lang_h, lang_hs = self.model.write(self.lang_h, self.ctx_h,
            100, self.args.temperature)
        self.logprobs.extend(logprobs)
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:'))
        self.words.append(outs)
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])
        return self._decode(outs, self.model.word_dict)

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, logprob, _ = self._choose(sample=True)
            self.logprobs.append(logprob)
        return choice

    def update(self, agree, reward, choice=None, partner_choice=None, partner_input=None, partner_reward=None):
        if not self.train:
            return

        self.t += 1
        if len(self.logprobs) == 0:
            return
        reward_agree = reward
        partner_reward_agree = partner_reward

        reward = reward if agree else 0
        partner_reward = partner_reward if agree else 0

        diff = reward - partner_reward
        self.all_rewards.append(diff)
        #self.all_rewards.append(reward)
        r = (diff - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        if self.args.visual and self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.loss_plot.update('loss', self.t, loss.data[0][0])

        self.opt.step()


class OnlineAgent(RnnRolloutAgent):
    def __init__(self, model, args, name='Alice'):
        super(OnlineAgent, self).__init__(model, args, name, force_unique_candidates=True, sample_selection=True,
                                          allow_no_agreement=False)
        self.eos = self.model.word_dict.get_idx('<eos>')
        self.eod = self.model.word_dict.get_idx('<selection>')
        self.t = 0
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr, #TODO
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        self.crit = Criterion(self.model.word_dict)
        self.sel_crit = Criterion(
            self.model.item_dict, bad_toks=['<disconnect>', '<disagree>'])
        self.last_choice = None
        self.sum_sel_out = None
        self.loss = 0
        self.lm_loss = 0

        self.choice_to_dialogue = dict()
        self.choices = set()
        self.agrees = 0
        self.reward = 0

        self.agreed_deal_to_dialogue = dict()

        self.no_agreements = 0

    def update(self, agree, reward, choice=None, partner_choice=None, partner_input=None):
        self.t += 1

        if choice[0] == '<no_agreement>':
            self.no_agreements += 1
            agree = True #FIXME

        assert(agree == (choice == partner_choice[3:] + partner_choice[:3]))

        data = torch.cat(self.words)
        words = self._decode(data, self.model.word_dict)

        self.choices.add(str(choice))

        if choice[0] != '<no_agreement>':
            # Disagree - adjudicate compromise
            # Compromise is the deal with the best minimum score, where each component agrees with at least one agent
            best_choice = None
            best_score = -99999

            for c in self.domain.generate_choices(self.context):
                match = True
                for i in range(0, len(c)):
                    j = (3 + i) % 6

                    if c[i] != choice[i] and c[i] != partner_choice[j]:
                        match = False
                        break
                if match:
                    score1 = self.domain.score(self.context, c)
                    score2 = self.domain.score(partner_input, c[3:] + c[:3])
                    score = min(score1, score2)

                    if score > best_score or (score == best_score and words[0] == 'YOU:'):
                        best_choice = c
                        best_score = score

            choice = best_choice
            partner_choice = best_choice[3:] + best_choice[:3]

        words_inverse = [('THEM:' if x == 'YOU:' else 'YOU:' if x == 'THEM:' else x) for x in words]

        if agree:
            self.agrees += 1
            self.reward += reward

        if self.t % 100 == 0:
            print ("Unique choices:    ", len(self.choices))
            print ("Agreements:        ", self.agrees)
            print ("No agreements:     ", self.no_agreements)
            print ("Reward:            ", self.reward)
            print ("Total uniq agreed: ", len(self.agreed_deal_to_dialogue))
            self.choices = set()
            self.reward = 0

            self.agrees = 0
            self.no_agreements = 0

        def insert(map, key, value):
            if not key in map:
                map[key] = []
            map[key].append(value)

        if choice[:3] == partner_choice[3:] and choice[0] != '<no_agreement>':
            insert(self.agreed_deal_to_dialogue, str(choice), (words, choice))
            insert(self.agreed_deal_to_dialogue, str(partner_choice), (words_inverse, partner_choice))

        if self.t % self.args.bsz == 0:
            # Train selection classifier
            # Take last N instances of each agreed deal
            deal_element_count = Counter()
            dialogue_to_deal = []
            for deal in self.agreed_deal_to_dialogue:
                deals_and_dialogues = self.agreed_deal_to_dialogue[deal]
                for (words, tgt_deal) in deals_and_dialogues[-10:]:
                    for i in range(0, len(tgt_deal)):
                        deal_element_count[(i, tgt_deal[i])] += 1

                    dialogue_to_deal.append((words, tgt_deal))

            loss = None
            loss_unweighted = 0

            num = 0

            # Classify deals
            for words, deal in dialogue_to_deal:
                sel_outs = self.model.forward_selection(Variable(self._encode(words, self.model.word_dict), requires_grad=False), None, None)
                for (i, deal_element) in enumerate(deal):
                    weight = 1 / deal_element_count[(i, deal_element)]

                    sel_out = sel_outs[i]
                    sel_tgt = Variable(torch.Tensor(self.model.item_dict.w2i([deal_element])).contiguous().view(-1).long(), volatile=False)

                    example_loss = self.sel_crit(sel_out.unsqueeze(0), sel_tgt)
                    #print (sel_out)
                    #print (sel_tgt)
                    #print (example_loss)
                    num += 1
                    loss_unweighted += example_loss
                    #print ("loss", loss, "ex", example_loss, "weight", weight)
                    loss = loss + example_loss * weight if loss is not None else example_loss * weight
                    #print ("new loss", loss)

            if self.t % 100 == 0:
                print ("selection loss", loss.data[0])
                print ("selection probability", math.exp(-loss_unweighted.data[0] / num))

            #print (loss)
            loss.backward()

            nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            self.opt.step()
            self.opt.zero_grad()

        # Train Language Model
        if self.args.lr_lm != 0.0:
            for words, ctx in [(data, self.ctx),
                               (self._encode(words_inverse, self.model.word_dict), self._encode(partner_input, self.model.context_dict))]:
                inpt = data.narrow(0, 0, words.size(0) - 1)
                tgt = data.narrow(0, 1, words.size(0) - 1).view(-1)

                out, _ = self.model.forward(inpt, Variable(ctx))
                self.lm_loss += self.crit(out, tgt) * self.args.lr_lm

            if self.t % 100 == 0:
                print("LM loss:           ", self.lm_loss.data[0])

            if self.t % self.args.bsz == 0:
                self.lm_loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
                self.opt.step()
                self.opt.zero_grad()
                self.lm_loss = 0

    def choose(self):
        choice, _, _ = self._choose(sample=True)
        return choice


class PredictionAgent(HierarchicalAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=False):
        super(PredictionAgent, self).__init__(model, args, name)
        self.train = train
        if self.train:
            self.model.train()
            self.opt = optim.RMSprop(
                self.model.pred_model.parameters(),
                lr=args.rl_lr,
                momentum=args.momentum)

        self.all_rewards = []
        self.acts = []
        self.logprobs = []
        self.entropys = []

        # kill sel_model in pred_model
        del self.model.pred_model.sel_model

        #if self.args.visual:
        if True:
            self.model_plot = vis.ModulePlot(self.model.pred_model, plot_weight=False, plot_grad=True)
            self.agree_plot = vis.Plot(['agree',], 'agree', 'agree')
            self.reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'reward', 'reward')
            self.loss_plot = vis.Plot(['loss',], 'loss', 'loss')
            self.agree_reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'agree_reward', 'agree_reward')
            self.entropy_plot = vis.Plot(['entropy',], 'entropy', 'entropy')
            self.temperature_plot = vis.Plot(['temperature',], 'temperature', 'temperature')
        self.t = 0

    def feed_context(self, context):
        self.sents = []
        self.context = context
        self.ctx = Variable(self._encode(context, dictionary=self.model.context_dict))
        self.ctx_h = self.model.ctx_encoder(self.ctx)
        self.sel_ctx_h = self.model.sel_model.ctx_encoder(self.ctx)
        self.pred_ctx_h = self.model.pred_model.ctx_encoder(self.ctx)
        self.lang_h = self.model.zero_h(1, self.model.args.nhid_lang).squeeze(0)
        self.sel_h = self.model.zero_h(1, self.model.args.nhid_lang).squeeze(0)
        self.update_state()
        self.plot_selection()
        self.acts = []
        self.logprobs = []
        self.entropys = []

    def choose(self):
        if not self.train or self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, logprob, _ = self._choose(sample=True)
            self.logprobs.append(logprob)
        return choice


    def update(self, agree, reward, choice=None, partner_choice=None, partner_input=None, partner_reward=None):
        if not self.train:
            return
        self.t += 1
        if len(self.logprobs) == 0:
            return
        reward_agree = reward
        partner_reward_agree = partner_reward

        reward = reward if agree else 0
        partner_reward = partner_reward if agree else 0

        diff = reward - partner_reward
        #self.all_rewards.append(diff)
        self.all_rewards.append(reward)
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        #r = diff
        #r = reward
        g = Variable(torch.zeros(1, 1).fill_(r))
        """
        loss = 0
        for lp in reversed(self.logprobs):
            loss = loss - lp * g
            g = g * self.args.gamma
        loss = loss / len(self.logprobs)
        """

        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        #if self.args.visual and self.t % 10 == 0:
        if self.t % 1 == 0:
            #self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.temperature_plot.update('temperature', self.t, self.args.pred_temperature)
            self.loss_plot.update('loss', self.t, loss.data[0][0])
            self.entropy_plot.update('entropy', self.t, np.mean(np.array(self.entropys)))
            #if self.t < 500:
            #    self.args.pred_temperature -= 1./50.0 * 9
        self.opt.step()

    def update2(self, agree, reward, choice=None, partner_choice=None, partner_input=None, partner_reward=None):
        if not self.train:
            return
        self.t += 1
        if len(self.acts) == 0:
            return
        reward_agree = reward
        partner_reward_agree = partner_reward

        reward = reward if agree else 0
        parther_reward = partner_reward if agree else 0
        self.all_rewards.append(reward)
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))

        for act in reversed(self.acts):
            R = torch.Tensor(1)
            R[0] = r
            act.reinforce(R)
            r *= self.args.gamma

        self.opt.zero_grad()
        autograd.backward(self.acts, [None for _ in self.acts])
        nn.utils.clip_grad_norm(self.model.pred_model.parameters(), self.args.rl_clip)
        #if self.args.visual and self.t % 10 == 0:
        if self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.temperature_plot.update('temperature', self.t, self.args.pred_temperature)
            self.entropy_plot.update('entropy', self.t, np.mean(np.array(self.entropys)))
            if self.t < 5000:
                self.args.pred_temperature -= self.t/5000.0 * 9
        self.opt.step()

    def _sample(self, prob, sample=True):
        if sample:
            return prob.multinomial().data[0][0]
        else:
            return torch.topk(prob, 1)[1].data[0][0]

    def update_state(self):
        lens, rev_idxs, hid_idxs = self._make_idxs(self.sents)

        sel_hs = self.model.sel_model.forward_inpts(self.sents, self.sel_ctx_h)
        pred_hs = self.model.pred_model.forward_inpts(self.sents, self.pred_ctx_h)

        self.sel_out, self.sel_sent_p, self.sel_word_ps = self.model.sel_model.sel_head(
            self.sel_ctx_h, sel_hs, lens, rev_idxs, hid_idxs)
        self.sel_out = [F.softmax(o) for o in self.sel_out]
        self.pred_cluster, self.pred_sent_p, self.pred_word_ps = self.model.pred_model.sel_head(
            self.pred_ctx_h, pred_hs, lens, rev_idxs, hid_idxs)
        self.pred_cluster = [F.softmax(o / self.args.pred_temperature) for o in self.pred_cluster[:1]]
        if self.train:
            prob = self.pred_cluster[0].squeeze(0)
            act = prob.multinomial()
            lp = prob.log().gather(0, act.detach())
            entropy = - (prob * prob.log()).sum()
            self.acts.append(act)
            self.logprobs.append(lp)
            self.entropys.append(entropy.data[0])
            idxs = [act.data[0]]
        else:
            idxs = [torch.topk(o, 1)[1].data[0][0] for o in self.pred_cluster]
        self.pred_out = self.model.dist_clusters.get_cluster_center_by_idx(
            self.ctx, idxs[0])
        self.pred_out = [Variable(o) for o in self.pred_out]

    def plot_selection(self, rl=False):
        choices = self.domain.generate_choices(self.context)

        def get_prob(dist, idxs):
            dist = dist.squeeze(0)
            dist = dist.gather(0, Variable(torch.Tensor(idxs).long()))
            return dist

        labels, idxs = [], []
        for choice in choices:
            idxs.append(self.model.item_dict.w2i(choice))
            if choice[0] == '<no_agreement>':
                labels.append('no_agree')
            elif choice[0] == '<disconnect>':
                labels.append('discon')
            else:
                you = [str(self.domain.parse_choice(c)[1]) for c in choice[:3]]
                them = [str(self.domain.parse_choice(c)[1]) for c in choice[3:]]
                labels.append('(%s)' % (','.join(you)))
        title = 'input %s' % (' '.join(['%s=(%s:%s)' % (DialogLogger.CODE2ITEM[i][1],
            self.context[2 * i + 0], self.context[2 * i + 1]) \
            for i in range(len(self.context) // 2)]))

        sel_prob = get_prob(self.sel_out[0], idxs)
        pred_prob = get_prob(self.pred_out[0], idxs)
        #pred_prob = get_prob(F.softmax(self.model.sel_model.sel_head.head(self.pred_out[0])), idxs)


        if self.vis:
            if rl:
                vis.plot_distribution(pred_prob, labels, 'RL move: ' + title)
            else:
                if len(self.sel_word_ps) > 0:
                    sents = [self._decode(s, self.model.word_dict) for s in self.sents]
                    #vis.plot_text(' '.join(sents[-1]))
                    #vis.plot_attn(sents, self.sel_sent_p, self.sel_word_ps)
                vis.plot_distribution(sel_prob, labels, 'your selection: ' + title)
                #vis.plot_distribution(sel_prob_inv, labels, 'their selection: ' + title)

                cluster_labels = ['c%d' % i for i in range(self.pred_cluster[0].size(1))]
                vis.plot_distribution(self.pred_cluster[0].squeeze(0), cluster_labels, 'clusters dist')
                val3, idx3 = torch.topk(self.pred_cluster[0].squeeze(0), 3)
                for i in range(3):
                    p = self.model.pred_model.dist_clusters.get_cluster_center_by_idx(self.ctx,
                        idx3.data[i])
                    vis.plot_distribution(get_prob(Variable(p[0]), idxs), labels, 'top%d: id=%d p=%.2f' % (
                        i, idx3.data[i], val3.data[i]))


                vis.plot_distribution(pred_prob, labels, 'your prediction: ' + title)
                #vis.plot_distribution(pred_prob_inv, labels, 'their prediction: ' + title)
            """
            input_pred = input('want to update prediciton? ')
            if input_pred != 'no':
                input_pred = [int(x) for x in input_pred.strip().split(' ')]
                key = ' '.join(['item%d=%d' % (i, x) for i, x in enumerate(input_pred)])
                cnts = self.domain.parse_context(self.context)[0]
                key_inv = ' '.join(['item%d=%d' % (i, c - x) for i, (x, c) in enumerate(zip(input_pred, cnts))])
                self.pred_out[0] = Variable(torch.Tensor(1, len(self.model.item_dict)).zero_())
                self.pred_out[0].data[0][self.model.item_dict.get_idx(key)] = 1
                self.pred_out[1] = Variable(torch.Tensor(1, len(self.model.item_dict)).zero_())
                self.pred_out[1].data[0][self.model.item_dict.get_idx(key_inv)] = 1
                vis.plot_distribution(get_prob(self.pred_out[0], idxs), labels, 'updated prediction: ' + title)
            """


            #vis.plot_distribution(pred_prob_inv, labels, 'their prediction: ' + title)
            #if attn_p.size(1) > 1:
            #    labels = ['%d' % i for i in range(attn_p.size(1))]
            #    #vis.plot_distribution(attn_p.squeeze(0), labels, 'Strategy Attention')
            #vis.plot_distribution(predicted_prob, labels, 'Predicted: ' + title)

            #score_out = F.softmax(self.model.forward_score(self.strat_hs, self.ctx_h))
            #labels = ['_%d' % i for i in range(11)]
            #vis.plot_distribution(score_out.squeeze(0), labels, 'Score')

    def read(self, inpt):
        if self.vis:
            import pdb; pdb.set_trace()

        inpt = ['THEM:'] + inpt
        # reverse for read
        pred_out = self.pred_out[::-1]

        inpt = Variable(self._encode(inpt, self.model.word_dict))
        self.sents.append(inpt)

        # read up the sentence
        self.lang_h, self.sel_h = self.model.read(inpt, self.ctx_h, pred_out, self.sel_h, self.lang_h)

        self.update_state()
        self.plot_selection()

    def _write(self):
        if self.vis:
            import pdb;
        return self._write()

    def _write_rollout(self, max_words=100):
        # don't need to reverse
        pred_out = self.pred_out

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        # don't need to reverse
        pred_out = self.pred_out

        # generate new sentence
        outs, self.lang_h, topks, self.sel_h, logprobs = self.model.write(
            self.lang_h, self.ctx_h, pred_out, self.sel_h, max_words, self.args.temperature)
        #self.logprobs.extend(logprobs)

        if self.vis:
            if outs.size(0) > 1:
                starts = ['YOU:'] + self.model.word_dict.i2w(outs.narrow(0, 0, outs.size(0) - 1).data[:, 0])
                for i in range(len(starts)):
                    start = starts[i]
                    prob, words = topks[i]
                    words = self.model.word_dict.i2w(words.data)
                    words = ['%s_' % w for w in words]
                    vis.plot_distribution(prob, words, start)

        self.sents.append(torch.cat([self.model.word2var('YOU:').unsqueeze(0), outs], 0))
        outs = self._decode(outs, self.model.word_dict)
        #inpt = ['YOU:'] + outs
        #inpt = Variable(self._encode(outs, self.model.word_dict))
        #lang_hs, self.lang_h = self.model.read(inpt, self.strat_h)

        self.update_state()
        self.plot_selection()

        return outs

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose(self, sample=False):
        print([' '.join(self._decode(s, self.model.word_dict)) for s in self.sents])

        if len(self.sents) > 1 and self._decode(self.sents[-2][1:], self.model.word_dict) == ['no', 'deal', '<eos>']:
            #TODO hack because apparently selection model never returns 'no deal'
            return ["<no_agreement>"] * 6, 0.0, 1.0

        choices = self.domain.generate_choices(self.context)

        sents = self.sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)

        sel_hs = self.model.sel_model.forward_inpts(sents, self.sel_ctx_h)
        self.sel_out, self.sel_sent_p, self.sel_word_ps = self.model.sel_model.sel_head(
            self.sel_ctx_h, sel_hs, lens, rev_idxs, hid_idxs)

        logit = self.sel_out[0].sub(self.sel_out[0].max(0)[0].data[0])
        prob = F.softmax(logit).squeeze(0)
        #_, idx = prob.max(0, keepdim=True)
        logprob = None

        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(logit.squeeze(0)).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None


        p_agree = prob[idx.data[0]].data[0]
        return self.model.item_dict.idx2word[idx.data[0]].split(' '), logprob, p_agree

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree


class PredictionRolloutAgent(PredictionAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=True):
        super(PredictionRolloutAgent, self).__init__(model, args, name)
        self.diverse = diverse

    def feed_partner_context(self, context_inv):
        pass

    def _sample_prediction(self, sents, pred_ctx_h, ctx, k=1):
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)

        pred_hs = self.model.pred_model.forward_inpts(sents, pred_ctx_h)
        cluster_dist_logit, _, _ = self.model.pred_model.sel_head(
            pred_ctx_h, pred_hs, lens, rev_idxs, hid_idxs)
        cluster_dist = F.softmax(cluster_dist_logit[0] / self.args.pred_temperature)
        cluster_idxs = cluster_dist.multinomial(k, replacement=not self.diverse).data[0]
        preds = []
        for i in range(k):
            pred = self.model.dist_clusters.get_cluster_center_by_idx(ctx, cluster_idxs[i])
            preds.append([Variable(p) for p in pred])
        return preds

    def _plot_cluster(self, context, center):
        choices = self.domain.generate_choices(self.context)
        labels, idxs = [], []
        for choice in choices:
            idxs.append(self.model.item_dict.w2i(choice))
            if choice[0] == '<no_agreement>':
                labels.append('no_agree')
            elif choice[0] == '<disconnect>':
                labels.append('discon')
            else:
                you = [str(self.domain.parse_choice(c)[1]) for c in choice[:3]]
                them = [str(self.domain.parse_choice(c)[1]) for c in choice[3:]]
                labels.append('(%s)' % (','.join(you)))
        title = 'input %s' % (' '.join(['%s=(%s:%s)' % (DialogLogger.CODE2ITEM[i][1],
            context[2 * i + 0], context[2 * i + 1]) for i in range(len(context) // 2)]))

        prob = center[0].squeeze(0).gather(0, Variable(torch.Tensor(idxs).long()))
        vis.plot_distribution(prob, labels, title)

    def _is_selection(self, out):
        return len(out) == 1 and self.model.word_dict.get_word(out.data[0][0]) == '<selection>'

    def _make_choice(self, sents, context, sel_ctx_h):
        if len(sents) > 1 and self._decode(sents[-2][1:], self.model.word_dict) == ['no', 'deal', '<eos>']:
            #TODO hack because apparently selection model never returns 'no deal'
            return ["<no_agreement>"] * 3, 0.0

        # remove selection
        sents = sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)

        sel_hs = self.model.sel_model.forward_inpts(sents, sel_ctx_h)
        sel_out, _, _ = self.model.sel_model.sel_head(
            sel_ctx_h, sel_hs, lens, rev_idxs, hid_idxs)
        sel_out = sel_out[0]

        logit = sel_out.sub(sel_out.max(0)[0].data[0])
        prob = F.softmax(logit).squeeze(0)
        _, idx = prob.max(0, keepdim=True)
        agree_prob = prob[idx.data[0]].data[0]
        choice = self.model.item_dict.idx2word[idx.data[0]].split(' ')
        #print('Choice: %s' % (' '.join(choice)))
        #print('Prob: %.2f' % agree_prob)
        return choice, agree_prob

    def _expected_score(self, sents):
        choice, prob = self._make_choice(sents, self.context, self.sel_ctx_h)
        score = self.domain.score(self.context, choice * 2)
        return score * prob

    def _rollout(self, pred, max_words):
        sents, lang_h, sel_h = self.sents[:], self.lang_h, self.sel_h

        YOU = self.model.word2var('YOU:').unsqueeze(0)
        THEM = self.model.word2var('THEM:').unsqueeze(0)

        # you write
        out2, lang_h2, _, sel_h2, _ = self.model.write(
            lang_h, self.ctx_h, pred, sel_h, max_words, self.args.temperature)
        score = 0.0
        n_samples = 5
        pred = self._sample_prediction(sents, self.pred_ctx_h, self.ctx, k=1)[0]

        for i in range(0, n_samples):
            sents = self.sents[:]
            out, lang_h, sel_h = out2, lang_h2, sel_h2

            while True:
                sents.append(torch.cat([YOU, out], 0))

                pred = self._sample_prediction(sents, self.pred_ctx_h, self.ctx, k=1)[0]

                if self._is_selection(out):
                    break
                # they write
                out, lang_h, _, sel_h, _ = self.model.write(
                    lang_h, self.ctx_h, pred[::-1], sel_h, max_words, self.args.temperature)
                sents.append(torch.cat([THEM, out], 0))

                # you read
                pred = self._sample_prediction(sents, self.pred_ctx_h, self.ctx, k=1)[0]

                if self._is_selection(out):
                    break

                # you write
                out, lang_h, _, sel_h, _ = self.model.write(
                    lang_h, self.ctx_h, pred, sel_h, max_words, self.args.temperature)

            score += self._expected_score(sents)
            print([' '.join(self._decode(s, self.model.word_dict)) for s in sents], self._expected_score(sents),
                  self._make_choice(sents, self.context, self.sel_ctx_h)[0])

        #print('Rollout:')
        #for s in sents:
        #    print(' '.join(self._decode(s, self.model.word_dict)))

        #print('Score: %.2f' % score)
        #import pdb; pdb.set_trace()
        return score / n_samples, sents[len(self.sents)]

    def _fanout(self, max_words, preds):
        #for pred in preds:
        #    self._plot_cluster(self.context, pred)
        #import pdb; pdb.set_trace()
        best_score, best_pred, best_sent, best_i = -1, None, None, None
        for i, pred in enumerate(preds):
            total_score = 0
            best_sent_for_pred = None
            best_score_for_pred = -1

            for _ in range(self.args.rollout_count_threshold):
                rollout_score, sent = self._rollout(pred, max_words)
                total_score += rollout_score
                if rollout_score > best_score_for_pred:
                    best_sent_for_pred, best_score_for_pred = sent, rollout_score


            #print('Score: %.3f' % (total_score / 3))
            if total_score > best_score:
                best_score = total_score
                best_pred = pred
                best_sent = best_sent_for_pred
                best_i = i

        return best_score / self.args.rollout_count_threshold, best_pred, best_sent

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        preds = self._sample_prediction(self.sents, self.pred_ctx_h, self.ctx, k=self.args.rollout_bsz)
        best_score, best_pred, best_sent = self._fanout(max_words, preds)



        outs = best_sent

        self.sents.append(outs)

        # read up the sentence
        self.lang_h, self.sel_h = self.model.read(outs, self.ctx_h, best_pred, self.sel_h, self.lang_h)

        outs = self._decode(outs[1:], self.model.word_dict)

        #print('Sentence: %s' % (' '.join(self._decode(self.sents[-1], self.model.word_dict))))
        #import pdb; pdb.set_trace()

        self.update_state()
        self.plot_selection()

        return outs


class LatentClusteringAgent(HierarchicalAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=False):
        super(LatentClusteringAgent, self).__init__(model, args, name)
        self.train = train
        if self.train:
            self.model.train()
            #self.model.clear_weights()
            self.opt = optim.RMSprop(
                self.model.parameters(),
                lr=args.rl_lr,
                momentum=args.momentum)
            if args.scratch:
                self.model.clear_weights()

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

        self.all_rewards = []
        self.acts = []
        self.logprobs = []
        self.entropys = []

        if train and args.validate:
            dom = domain.get_domain(args.domain)
            corpus = self.model.corpus_ty(dom, args.data, freq_cutoff=args.unk_threshold,
                verbose=True, sep_sel=self.model.args.sep_sel)
            self.validset, self.validset_stats = corpus.valid_dataset(args.bsz)
            self.engine = self.model.engine_ty(self.model, args)

            self.last_ppl = np.exp(self._validate_model())

        if self.train and self.args.visual:
            prefix = 'rl_data/pretrained/seed_%d' % self.args.seed
            self.model_plot = vis.ModulePlot(self.model,
                plot_weight=True, plot_grad=False)
            self.agree_plot = vis.Plot(['agree',], 'agree', 'agree', prefix=prefix)
            self.advantage_plot = vis.Plot(['advantage','max_advantage'], 'advantage', 'advantage',
                prefix=prefix)
            self.reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'reward', 'reward', prefix=prefix)
            self.loss_plot = vis.Plot(['loss',], 'loss', 'loss', prefix=prefix)
            self.max_reward_plot = vis.Plot(
                ['reward', 'partner_reward'], 'max_reward', 'max_reward', prefix=prefix)
            self.entropy_plot = vis.Plot(['entropy',], 'entropy', 'entropy', prefix=prefix)
            self.ppl_plot = vis.Plot(['ppl',], 'ppl', 'ppl', prefix=prefix)
            self.temperature_plot = vis.Plot(['temperature',], 'temperature', 'temperature',
                prefix=prefix)
        self.t = 0

    def feed_context(self, context):
        self.sents = []
        self.logprobs = []
        self.entropys = []
        self.context = context
        self.ctx = Variable(self._encode(context, dictionary=self.model.context_dict))
        self.cnt = Variable(torch.Tensor([self.model.count_dict.get_idx(context)]).long())
        # init model
        self.ctx_h = self.model.ctx_encoder(self.ctx)
        self.lang_enc_h = self.model._zero(1, self.model.lang_model.args.nhid_lang)
        self.mem_h = self.model.forward_memory(self.ctx_h, mem_h=None, inpt=None)

    def choose(self):
        if not self.train or self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, logprob, _ = self._choose(sample=True)
            self.logprobs.append(logprob)
        return choice

    def _validate_model(self, volatile=True):
        loss, _, _, _ = self.engine.valid_pass(self.validset, self.validset_stats)
        return loss

    def read(self, inpt):
        inpt = ['THEM:'] + inpt
        inpt = Variable(self._encode(inpt, self.model.word_dict))
        self.sents.append(inpt)
        self.lang_enc_h, self.mem_h = self.model.read(inpt, self.lang_enc_h, self.mem_h, self.ctx_h)

    def write(self, max_words=100):
        _, lat_h, log_q_z = self.model.forward_prediction(self.cnt, self.mem_h, sample=self.train)
        if self.train:
            self.logprobs.append(log_q_z)
            self.entropys.append(-(log_q_z.exp() * log_q_z).sum().data[0])

        out, logprobs = self.model.write(self.lang_enc_h, lat_h, max_words, self.args.temperature)
        #self.logprobs += logprobs
        self.sents.append(out)
        self.lang_enc_h, self.mem_h = self.model.read(out, self.lang_enc_h, self.mem_h, self.ctx_h)

        return self._decode(out[1:], self.model.word_dict)

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def update(self, agree, max_reward, choice=None, partner_choice=None,
        partner_input=None, max_partner_reward=None):

        if not self.train:
            return
        self.t += 1

        if len(self.logprobs) == 0:
            return

        reward = max_reward if agree else 0
        partner_reward = max_partner_reward if agree else 0
        self.all_rewards.append(reward)
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))

        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()

        # don't update clusters
        self.model.latent_bottleneck.zero_grad()
        # don't update language model
        self.model.lang_model.zero_grad()

        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        self.opt.step()

        if self.args.visual and self.t % 10 == 0:
            #self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.max_reward_plot.update('reward', self.t, max_reward)
            self.max_reward_plot.update('partner_reward', self.t, max_partner_reward)
            self.advantage_plot.update('advantage', self.t, reward - partner_reward)
            self.advantage_plot.update('max_advantage', self.t, max_reward - max_partner_reward)
            #self.temperature_plot.update('temperature', self.t, self.args.pred_temperature)
            self.loss_plot.update('loss', self.t, loss.data[0][0])
            self.entropy_plot.update('entropy', self.t, np.mean(np.array(self.entropys)))
            if self.t % 100 == 0:
                self.last_ppl = np.exp(self._validate_model())
            self.ppl_plot.update('ppl', self.t, self.last_ppl)
            #if self.t < 500:
            #    self.args.pred_temperature -= 1./50.0 * 9

    def _choose(self, sample=False):
        sents = self.sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, self.ctx)

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].item())
        prob = F.softmax(choice_logit, dim=0)

        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.item()]

        # Pick only your choice
        return choices[idx.item()][:self.domain.selection_length()], logprob, p_agree.item()


class LatentClusteringRolloutAgent(LatentClusteringAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=True):
        super(LatentClusteringRolloutAgent, self).__init__(model, args, name)
        self.diverse = diverse

    def feed_partner_context(self, context_inv):
        pass

    def _is_selection(self, out):
        return len(out) == 2 and self.model.word_dict.get_word(out.data[1][0]) == '<selection>'

    def _make_choice(self, sents, context):
        sents = sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, self.ctx)

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        idx = prob.multinomial().detach()

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]], p_agree.data[0]

    def _expected_score(self, sents):
        choice, prob = self._make_choice(sents, self.context)
        score = self.domain.score(self.context, choice)
        return score * prob

    def _rollout(self, lang_enc_h, lat_h, mem_h, max_words, n_samples=3):
        score = 0
        sent_start, _ = self.model.write(lang_enc_h, lat_h, max_words,
            self.args.temperature)
        start_lang_enc_h, start_mem_h = self.model.read(sent_start,
            lang_enc_h, mem_h, self.ctx_h)
        #print(' '.join(self._decode(sent_start, self.model.word_dict)))

        for i in range(0, n_samples):
            # you write
            sents = [sent_start]
            lang_enc_h, mem_h = start_lang_enc_h, start_mem_h

            if self._is_selection(sent_start):
                score += self._expected_score(self.sents + sents)
                break

            while True:
                # they write
                _, lat_h, _ = self.model.forward_prediction(
                    self.cnt, mem_h, sample=False)
                sent, _ = self.model.write(lang_enc_h, lat_h, max_words,
                    self.args.temperature, start_token='THEM:')
                sents.append(sent)
                lang_enc_h, mem_h = self.model.read(sent, lang_enc_h, mem_h, self.ctx_h)

                if self._is_selection(sent):
                    break

                # you write
                _, lat_h, _ = self.model.forward_prediction(
                    self.cnt, mem_h, sample=True)
                sent, _ = self.model.write(lang_enc_h, lat_h, max_words,
                    self.args.temperature, start_token='YOU:')
                sents.append(sent)
                lang_enc_h, mem_h = self.model.read(sent, lang_enc_h, mem_h, self.ctx_h)

                if self._is_selection(sent):
                    break

            score += self._expected_score(self.sents + sents)
        return score / (i + 1), sents[0]

    def _fanout(self, lang_enc_h, mem_h, max_words):
        num_samples = self.args.rollout_bsz
        zs, lat_hs = self.model.forward_prediction_multi(self.cnt, self.mem_h,
            num_samples=num_samples, sample=not self.diverse)

        best_score, best_sent = -1, None
        for i in range(num_samples):
            # take i-th sample
            lat_h = lat_hs.narrow(0, i, 1)
            #print(zs.narrow(1, i, 1).data[0][0])

            total_score, best_cur_score, best_cur_sent = 0, -1, None
            for _ in range(self.args.rollout_count_threshold):
                score, sent = self._rollout(lang_enc_h, lat_h, mem_h, max_words)
                total_score += score
                if score > best_cur_score:
                    best_cur_score, best_cur_sent = score, sent

            if total_score > best_score:
                best_score = total_score
                best_sent = best_cur_sent

            #import pdb; pdb.set_trace()

        best_score /= self.args.rollout_count_threshold
        return best_score, best_sent

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        _, out = self._fanout(self.lang_enc_h, self.mem_h, max_words)

        self.sents.append(out)
        self.lang_enc_h, self.mem_h = self.model.read(
            out, self.lang_enc_h, self.mem_h, self.ctx_h)

        return self._decode(out[1:], self.model.word_dict)


class LatentClusteringFastRolloutAgent(LatentClusteringAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=True):
        super(LatentClusteringFastRolloutAgent, self).__init__(model, args, name)
        self.diverse = diverse

    def set_rollout_model(self, rollout_model):
        self.rollout_model = rollout_model

    def feed_partner_context(self, context_inv):
        pass

    def feed_context(self, context):
        super(LatentClusteringFastRolloutAgent, self).feed_context(context)
        #self.rollout_h = self.rollout_model.forward_ctx(self.ctx)
        self.rollout_mem_h = self.rollout_model.init_memory(self.ctx)
        # run first step
        #_, self.rollout_mem_h, _ = self.rollout_model.step(
        #    self.rollout_model._zero(1, self.rollout_model.pred_model.lang_model.cluster_model.args.nhid_cluster),
        #    self.rollout_mem_h, self.cnt)

    def _make_choice(self, sel_out, context):
        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        p_agree, idx = torch.topk(prob, 1)

        # Pick only your choice
        return choices[idx.data[0]], p_agree.data[0]

    def _expected_score(self, sel_out):
        choice, prob = self._make_choice(sel_out, self.context)
        score = self.domain.score(self.context, choice)
        return score * prob

    def _rollout(self, lang_enc_h, lat_h, mem_h, max_words, n_samples=1):
        score = 0
        sent_start, _ = self.model.write(lang_enc_h, lat_h, max_words,
            self.args.temperature)
        start_lang_enc_h, start_mem_h = self.model.read(sent_start,
            lang_enc_h, mem_h, self.ctx_h)
        _, start_lat_h, _ = self.model.forward_prediction(self.cnt, start_mem_h, sample=False)
        for i in range(0, n_samples):
            sel_out = self.rollout_model.rollout(start_lat_h, self.rollout_mem_h, self.cnt)
            score += self._expected_score(sel_out)
        return score, sent_start

    def _fanout(self, lang_enc_h, mem_h, max_words):
        num_samples = self.args.rollout_bsz
        zs, lat_hs = self.model.forward_prediction_multi(self.cnt, self.mem_h,
            num_samples=num_samples, sample=not self.diverse)

        best_score, best_sent, best_lat_h = -1, None, None
        for i in range(num_samples):
            # take i-th sample
            lat_h = lat_hs.narrow(0, i, 1)

            total_score, best_cur_score, best_cur_sent = 0, -1, None
            for _ in range(self.args.rollout_count_threshold):
                score, sent = self._rollout(lang_enc_h, lat_h, mem_h, max_words)
                #print('score: %.2f' % score)
                #print('sent: %s' % ' '.join(self._decode(sent, self.model.word_dict)))
                total_score += score
                if score > best_cur_score:
                    best_cur_score, best_cur_sent = score, sent

            if total_score > best_score:
                best_score = total_score
                best_sent = best_cur_sent
                best_lat_h = lat_h

        #import pdb; pdb.set_trace()
        best_score /= self.args.rollout_count_threshold
        return best_score, best_sent, best_lat_h

    def read(self, inpt):
        super(LatentClusteringFastRolloutAgent, self).read(inpt)
        _, lat_h, _ = self.model.forward_prediction(self.cnt, self.mem_h, sample=False)
        _, self.rollout_mem_h, _ = self.rollout_model.step(lat_h, self.rollout_mem_h, self.cnt)

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        _, out, lat_h = self._fanout(self.lang_enc_h, self.mem_h, max_words)

        self.sents.append(out)
        self.lang_enc_h, self.mem_h = self.model.read(
            out, self.lang_enc_h, self.mem_h, self.ctx_h)
        # update rollout model state
        _, self.rollout_mem_h, _ = self.rollout_model.step(lat_h, self.rollout_mem_h, self.cnt)

        return self._decode(out[1:], self.model.word_dict)


class LatentClusteringFasterRolloutAgent(LatentClusteringAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=True):
        super(LatentClusteringFasterRolloutAgent, self).__init__(model, args, name)
        self.diverse = diverse

    def set_rollout_model(self, rollout_model):
        self.rollout_model = rollout_model

    def feed_partner_context(self, context_inv):
        pass

    def feed_context(self, context):
        super(LatentClusteringFasterRolloutAgent, self).feed_context(context)
        self.cluster_mem_h = self.rollout_model._zero(1, self.rollout_model.args.nhid_lang)

    def _make_choice(self, sel_out, context):
        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        idx = prob.multinomial().detach()

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]], p_agree.data[0]

    def _expected_score(self, sel_out):
        choice, prob = self._make_choice(sel_out, self.context)
        score = self.domain.score(self.context, choice)
        return score * prob

    def _rollout(self, lang_enc_h, lat_h, mem_h, max_words):
        sent_start, _ = self.model.write(lang_enc_h, lat_h, max_words,
            self.args.temperature)

        cluster_mem_h = self.rollout_model.cluster_model.memory(lat_h, self.cluster_mem_h)
        sel_out = self.rollout_model.selection(cluster_mem_h)
        score = self._expected_score(sel_out)
        return score, sent_start

    def _fanout(self, lang_enc_h, mem_h, max_words):
        num_samples = self.args.rollout_bsz
        zs, lat_hs = self.model.forward_prediction_multi(self.cnt, self.mem_h,
            num_samples=num_samples, sample=not self.diverse)

        best_score, best_sent, best_lat_h = -1, None, None
        for i in range(num_samples):
            # take i-th sample
            lat_h = lat_hs.narrow(0, i, 1)

            total_score, best_cur_score, best_cur_sent = 0, -1, None
            for _ in range(self.args.rollout_count_threshold):
                score, sent = self._rollout(lang_enc_h, lat_h, mem_h, max_words)
                total_score += score
                if score > best_cur_score:
                    best_cur_score, best_cur_sent = score, sent

            if total_score > best_score:
                best_score = total_score
                best_sent = best_cur_sent
                best_lat_h = lat_h

        best_score /= self.args.rollout_count_threshold
        return best_score, best_sent, best_lat_h

    def read(self, inpt):
        super(LatentClusteringFasterRolloutAgent, self).read(inpt)
        _, lat_h, _ = self.model.forward_prediction(self.cnt, self.mem_h, sample=False)
        self.cluster_mem_h = self.rollout_model.cluster_model.memory(lat_h, self.cluster_mem_h)

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        _, out, lat_h = self._fanout(self.lang_enc_h, self.mem_h, max_words)

        self.sents.append(out)
        self.lang_enc_h, self.mem_h = self.model.read(
            out, self.lang_enc_h, self.mem_h, self.ctx_h)
        # update rollout model state
        self.cluster_mem_h = self.rollout_model.cluster_model.memory(lat_h, self.cluster_mem_h)

        return self._decode(out[1:], self.model.word_dict)


class BaselineClusteringAgent(HierarchicalAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=False):
        super(BaselineClusteringAgent, self).__init__(model, args, name)
        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()

    def feed_context(self, context):
        self.sents = []
        self.context = context
        print(context)
        self.ctx = Variable(self._encode(context, dictionary=self.model.context_dict))
        self.cnt = Variable(torch.Tensor([self.model.count_dict.get_idx(context)]).long())
        # init model
        self.ctx_h = self.model.ctx_encoder(self.ctx)
        self.mem_h = self.model.init_memory(self.ctx_h)

    def choose(self):
        choice, _, _ = self._choose(sample=False)
        return choice

    def read(self, inpt):
        inpt = ['THEM:'] + inpt
        inpt = Variable(self._encode(inpt, self.model.word_dict))
        self.sents.append(inpt)
        self.mem_h = self.model.read(inpt, self.mem_h)

    def write(self, max_words=100):
        lat_h, _ = self.model.latent_bottleneck(self.cnt, self.mem_h)

        out = self.model.write(lat_h, max_words, self.args.temperature)
        self.sents.append(out)
        self.mem_h = self.model.read(out, self.mem_h)

        return self._decode(out[1:], self.model.word_dict)

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def update(self, agree, max_reward, choice=None, partner_choice=None,
        partner_input=None, max_partner_reward=None):
        pass

    def _choose(self, sample=False):
        sents = self.sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, self.ctx)

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        if sample:
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree.data[0]


class BaselineClusteringRolloutAgent(BaselineClusteringAgent):
    def __init__(self, model, args, name='Alice', train=False, diverse=True):
        super(BaselineClusteringRolloutAgent, self).__init__(model, args, name)
        self.diverse = diverse

    def feed_partner_context(self, context_inv):
        pass

    def _is_selection(self, out):
        return len(out) == 2 and self.model.word_dict.get_word(out.data[1][0]) == '<selection>'

    def _make_choice(self, sents, context):
        sents = sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, self.ctx)

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].data[0])
        prob = F.softmax(choice_logit)

        idx = prob.multinomial().detach()

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]], p_agree.data[0]

    def _expected_score(self, sents):
        choice, prob = self._make_choice(sents, self.context)
        score = self.domain.score(self.context, choice)
        return score * prob

    def _rollout(self, lat_h, mem_h, max_words, n_samples=1):
        score = 0
        sent_start = self.model.write(lat_h, max_words,
            self.args.temperature)
        print(' '.join(self._decode(sent_start, self.model.word_dict)))
        return 0, 0
        start_lang_enc_h, start_mem_h = self.model.read(sent_start,
            lang_enc_h, mem_h, self.ctx_h)
        #print(' '.join(self._decode(sent_start, self.model.word_dict)))

        for i in range(0, n_samples):
            # you write
            sents = [sent_start]
            lang_enc_h, mem_h = start_lang_enc_h, start_mem_h

            if self._is_selection(sent_start):
                score += self._expected_score(self.sents + sents)
                break

            while True:
                # they write
                _, lat_h, _ = self.model.forward_prediction(
                    self.cnt, mem_h, sample=False)
                sent, _ = self.model.write(lang_enc_h, lat_h, max_words,
                    self.args.temperature, start_token='THEM:')
                sents.append(sent)
                lang_enc_h, mem_h = self.model.read(sent, lang_enc_h, mem_h, self.ctx_h)

                if self._is_selection(sent):
                    break

                # you write
                _, lat_h, _ = self.model.forward_prediction(
                    self.cnt, mem_h, sample=True)
                sent, _ = self.model.write(lang_enc_h, lat_h, max_words,
                    self.args.temperature, start_token='YOU:')
                sents.append(sent)
                lang_enc_h, mem_h = self.model.read(sent, lang_enc_h, mem_h, self.ctx_h)

                if self._is_selection(sent):
                    break

            score += self._expected_score(self.sents + sents)
        return score / (i + 1), sents[0]

    def _fanout(self, mem_h, max_words):
        num_samples = self.args.rollout_bsz

        _, p_z = self.model.latent_bottleneck(self.cnt, mem_h)
        _, zs = torch.topk(p_z, num_samples)
        lat_hs = self.model.latent_bottleneck.select(self.cnt, zs)

        best_score, best_sent = -1, None
        for i in range(0, num_samples):
            # take i-th sample
            lat_h = lat_hs.narrow(0, i, 1)
            print(zs.narrow(1, i, 1).data[0][0])

            total_score, best_cur_score, best_cur_sent = 0, -1, None
            for _ in range(0, self.args.rollout_count_threshold):
                score, sent = self._rollout(lat_h, mem_h, max_words)
                total_score += score
                if score > best_cur_score:
                    best_cur_score, best_cur_sent = score, sent

            if total_score > best_score:
                best_score = total_score
                best_sent = best_cur_sent

            import pdb; pdb.set_trace()

        best_score /= self.args.rollout_count_threshold
        return best_score, best_sent

    def write(self, max_words=100):
        if self.vis:
            import pdb; pdb.set_trace()

        _, out = self._fanout(self.mem_h, max_words)

        self.sents.append(out)
        self.mem_h = self.model.read(out, self.mem_h)

        return self._decode(out[1:], self.model.word_dict)






"""


class HumanAgent(Agent):
    def __init__(self, domain, name='Human'):
        self.name = name
        self.human = True
        self.domain = domain

    def feed_context(self, ctx):
        self.ctx = ctx

    def feed_partner_context(self, partner_context):
        pass

    def write(self):
        while True:
            try:
                return input('%s : ' % self.name).lower().strip().split() + ['<eos>']
            except KeyboardInterrupt:
                sys.exit()
            except:
                print('Your sentence is invalid! Try again.')

    def choose(self):
        while True:
            try:
                choice = input('%s choice: ' % self.name)
                return self.domain.parse_human_choice(self.ctx, choice)
            except KeyboardInterrupt:
                sys.exit()
            #except:
            #    print('Your choice is invalid! Try again.')

"""
