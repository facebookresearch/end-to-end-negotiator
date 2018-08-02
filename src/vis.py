# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A visualization library. Relies on visdom.
"""

import pdb

import visdom
import numpy as np


class Plot(object):
    """A class for plotting and updating the plot in real time."""
    def __init__(self, metrics, title, ylabel, xlabel='t', running_n=100):
        self.vis = visdom.Visdom()
        self.metrics = metrics
        self.opts = dict(
            fillarea=False,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        self.win = None
        self.running_n = running_n
        self.vals = dict()
        self.cnts = dict()

    def _update_metric(self, metric, x, y):
        if metric not in self.vals:
            self.vals[metric] = np.zeros(self.running_n)
            self.cnts[metric] = 0

        self.vals[metric][self.cnts[metric] % self.running_n] = y
        self.cnts[metric] += 1

        y = self.vals[metric][:min(self.cnts[metric], self.running_n)].mean()
        return np.array([x]), np.array([y])

    def update(self, metric, x, y):
        assert metric in self.metrics, 'metric %s is not in %s' % (metric, self.metrics)
        X, Y = self._update_metric(metric, x, y)
        if self.win is None:
            self.opts['legend'] = [metric,]
            self.win = self.vis.line(X=X, Y=Y, opts=self.opts)
        else:
            self.vis.line(X=X, Y=Y, win=self.win, update='append', name=metric)


class ModulePlot(object):
    """A helper class that plots norms of weights and gradients for a given module."""
    def __init__(self, module, plot_weight=False, plot_grad=False, running_n=100):
        self.module = module
        self.plot_weight = plot_weight
        self.plot_grad = plot_grad
        self.plots = dict()

        def make_plot(m, n):
            names = m._parameters.keys()
            if self.plot_weight:
                self.plots[n + '_w'] = Plot(names, n + '_w', 'norm', running_n=running_n)
            if self.plot_grad:
                self.plots[n + '_g'] = Plot(names, n + '_g', 'norm', running_n=running_n)

        self._for_all(make_plot, self.module)

    def _for_all(self, fn, module, name=None):
        name = name or module.__class__.__name__.lower()
        if len(module._modules) == 0:
            fn(module, name)
        else:
            for n, m in module._modules.items():
                self._for_all(fn, m, name + '_' + n)

    def update(self, x):
        def update_plot(m, n):
            for k, p in m._parameters.items():
                if self.plot_weight:
                    self.plots[n + '_w'].update(k, x, p.norm().item())
                if self.plot_grad and hasattr(p, 'grad') and p.grad is not None:
                    self.plots[n + '_g'].update(k, x, p.grad.norm().item())

        self._for_all(update_plot, self.module)
