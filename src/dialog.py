# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pdb

import numpy as np

from metric import MetricsContainer
import data
import utils
import domain


class DialogLogger(object):
    CODE2ITEM = [
        ('item0', 'book'),
        ('item1', 'hat'),
        ('item2', 'ball'),
    ]

    def __init__(self, verbose=False, log_file=None, append=False):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def dump_ctx(self, name, ctx):
        assert len(ctx) == 6, 'we expect 3 objects'
        s = ' '.join(['%s=(count:%s value:%s)' % (self.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1]) \
            for i in range(3)])
        self._dump_with_name(name, s)

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_choice(self, name, choice):
        def rep(w):
            p = w.split('=')
            if len(p) == 2:
                for k, v in self.CODE2ITEM:
                    if p[0] == k:
                        return '%s=%s' % (v, p[1])
            return w

        self._dump_with_name(name, ' '.join([rep(c) for c in choice]))

    def dump_agreement(self, agree):
        self._dump('Agreement!' if agree else 'Disagreement?!')

    def dump_reward(self, name, agree, reward):
        if agree:
            self._dump_with_name(name, '%d points' % reward)
        else:
            self._dump_with_name(name, '0 (potential %d)' % reward)

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])

    def dump_reward(self, name, agree, reward):
        pass


class Dialog(object):
    def __init__(self, agents, args):
        # For now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_moving_percentage('moving_agree')
        self.metrics.register_average('advantage')
        self.metrics.register_moving_average('moving_advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        self.metrics.register_average('agree_comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_moving_average('%s_moving_rew' % agent.name)
            self.metrics.register_average('agree_%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        if self.args.ref_text:
            ref_text = ' '.join(data.read_lines(self.args.ref_text))
            self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and (out[0] in ['<selection>', '<no_agreement>'])

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger, max_words=5000):
        assert len(self.agents) == len(ctxs)
        for agent, ctx, partner_ctx in zip(self.agents, ctxs, reversed(ctxs)):
            agent.feed_context(ctx)
            agent.feed_partner_context(partner_ctx)
            logger.dump_ctx(agent.name, ctx)
        logger.dump('-' * 80)

        # Choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        self.metrics.reset()

        #words_left = np.random.randint(50, 200)
        words_left = max_words
        length = 0
        expired = False

        while True:
            out = writer.write(max_words=words_left)
            words_left -= len(out)
            length += len(out)

            self.metrics.record('sent_len', len(out))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            conv.append(out)
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)

            if self._is_selection(out):
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break

            if words_left <= 1:
                break

            writer, reader = reader, writer


        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)
            logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])

        agree, rewards = self.domain.score_choices(choices, ctxs)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            logger.dump_reward(agent.name, agree, reward)
            j = 1 if i == 0 else 0
            agent.update(agree, reward, choice=choices[i],
                partner_choice=choices[j], partner_input=ctxs[j], max_partner_reward=rewards[j])

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for agent, reward in zip(self.agents, rewards):
                self.metrics.record('agree_%s_rew' % agent.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards
