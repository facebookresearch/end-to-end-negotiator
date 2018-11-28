# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Configuration script. Stores variables and settings used across application
"""

import logging

log_level = logging.INFO
log_format = '%(asctime)s : %(levelname)s : %(filename)s : %(message)s'

# default training settings
data_dir = 'data/negotiate' # data corpus directory
nembed_word = 256 # size of word embeddings
nembed_ctx = 64 # size of context embeddings
nhid_lang = 256 # size of the hidden state for the language model
nhid_ctx = 64 # size of the hidden state for the context model
nhid_strat = 64 # size of the hidden state for the strategy model
nhid_attn = 64 # size of the hidden state for the attention module
nhid_sel = 64 # size of the hidden state for the selection module
lr = 20.0 # initial learning rate
min_lr = 1e-5 # min thresshold for learning rate annealing
decay_rate = 9.0 # decrease learning rate by this factor
decay_every = 1 # decrease learning rate after decay_every epochs
momentum = 0.0 # momentum for SGD
nesterov = False # enable Nesterov momentum
clip = 0.2 # gradient clipping
dropout = 0.5 # dropout rate in embedding layer
init_range = 0.1 #initialization range
max_epoch = 30 # max number of epochs
bsz = 25 # batch size
unk_threshold = 20 # minimum word frequency to be in dictionary
temperature = 0.1 # temperature
sel_weight = 1.0 # selection weight
seed = 1 # random seed
cuda = False # use CUDA
plot_graphs = False # use visdom
domain = "object_division" # domain for the dialogue
rnn_ctx_encoder = False # Whether to use RNN for encoding the context

#fixes
rl_gamma = 0.95
rl_eps = 0.0
rl_momentum = 0.0
rl_lr = 20.0 
rl_clip = 0.2
rl_reinforcement_lr = 20.0
rl_reinforcement_clip = 0.2
rl_bsz = 25
rl_sv_train_freq = 4
rl_nepoch = 4
rl_score_threshold= 6
verbose = True
rl_temperature = 0.1
