# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable

from engines import EngineBase, Criterion


class SelectionEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(SelectionEngine, self).__init__(model, args, verbose)
        self.sel_crit = Criterion(
            self.model.item_dict,
            bad_toks=['<disconnect>', '<disagree>'],
            reduction='mean' if args.sep_sel else 'none')

    def _forward(model, batch, sep_sel=False):
        ctx, _, inpts, lens, _, sel_tgt, rev_idxs, hid_idxs, _ = batch
        ctx = Variable(ctx)
        inpts = [Variable(inpt) for inpt in inpts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]
        if sep_sel:
            sel_tgt = Variable(sel_tgt)
        else:
            sel_tgt = [Variable(t) for t in sel_tgt]

        # remove YOU:/THEM: from the end
        sel_out = model(inpts[:-1], lens[:-1], rev_idxs[:-1], hid_idxs[:-1], ctx)

        return sel_out, sel_tgt

    def train_batch(self, batch):
        sel_out, sel_tgt = SelectionEngine._forward(self.model, batch,
            sep_sel=self.args.sep_sel)
        loss = 0
        if self.args.sep_sel:
            loss = self.sel_crit(sel_out, sel_tgt)
        else:
            for out, tgt in zip(sel_out, sel_tgt):
                loss += self.sel_crit(out, tgt)
            loss /= sel_out[0].size(0)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return loss.item()

    def valid_batch(self, batch):
        with torch.no_grad():
            sel_out, sel_tgt = SelectionEngine._forward(self.model, batch,
                sep_sel=self.args.sep_sel)
        loss = 0
        if self.args.sep_sel:
            loss = self.sel_crit(sel_out, sel_tgt)
        else:
            for out, tgt in zip(sel_out, sel_tgt):
                loss += self.sel_crit(out, tgt)
            loss /= sel_out[0].size(0)

        return 0, loss.item(), 0


