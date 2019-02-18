# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable

from engines import EngineBase, Criterion


class RnnEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(RnnEngine, self).__init__(model, args, verbose)

    def _forward(model, batch):
        ctx, inpt, tgt, sel_tgt = batch
        ctx = Variable(ctx)
        inpt = Variable(inpt)
        tgt = Variable(tgt)
        sel_tgt = Variable(sel_tgt)

        out, sel_out = model(inpt, ctx)
        return out, tgt, sel_out, sel_tgt

    def train_batch(self, batch):
        out, tgt, sel_out, sel_tgt = RnnEngine._forward(self.model, batch)
        loss = self.crit(out, tgt)
        loss += self.sel_crit(sel_out, sel_tgt) * self.model.args.sel_weight
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return loss.item()

    def valid_batch(self, batch):
        with torch.no_grad():
            out, tgt, sel_out, sel_tgt = RnnEngine._forward(self.model, batch)
        valid_loss = tgt.size(0) * self.crit(out, tgt)
        select_loss = self.sel_crit(sel_out, sel_tgt)
        return valid_loss.item(), select_loss.item(), 0
