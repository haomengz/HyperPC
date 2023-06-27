# @file decoder_topnet.py
# @author Junming Zhang, junming@umich.edu
# @brief topnet decoder
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import math
import numpy as np
import torch.nn.functional as F
from copy import deepcopy


# Number of children per tree levels for 2048 output points
tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = deepcopy(tree_arch[nlevels])
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch


class MLP(torch.nn.Module):
    def __init__(self, dims,  bn=None):
        super().__init__()
        self.model = torch.nn.Sequential()
        self.bn = bn
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('fc_%d' % (i+1), torch.nn.Linear(num_channels, dims[i+1]))
            if self.bn:
                self.model.add_module('bn_%d' % (i+1), torch.nn.BatchNorm1d(dims[i+1]))
            self.model.add_module('relu_%d' % (i+1), torch.nn.ReLU())

        self.output_layer = torch.nn.Linear(dims[-2], dims[-1])

    def forward(self, features):
        features = self.model(features)
        outputs = self.output_layer(features)
        return outputs


class MLPConv(torch.nn.Module):
    def __init__(self, dims, bn=None):
        super().__init__()
        self.model = torch.nn.Sequential()
        self.bn = bn
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('conv1d_%d' % (i+1), torch.nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
            if self.bn:
                self.batch_norm = torch.nn.BatchNorm1d(dims[i+1])
            self.model.add_module('relu_%d' % (i+1), torch.nn.ReLU())

        self.output_layer = torch.nn.Conv1d(dims[-2], dims[-1], kernel_size=1)

    def forward(self, inputs):
        inputs = self.model(inputs)
        outputs = self.output_layer(inputs)
        return outputs


class CreateLevel(torch.nn.Module):
    def __init__(self, level, input_channels, output_channels, bn, tarch):
        super().__init__()
        self.output_channels = output_channels
        self.mlp_conv = MLPConv([input_channels, input_channels, int(input_channels / 2), int(input_channels / 4),
                                 int(input_channels / 8), output_channels * int(tarch[level])], bn=bn)

    def forward(self, inputs):
        features = self.mlp_conv(inputs)
        features = features.view(features.shape[0], self.output_channels, -1)
        return features


class TopNetDecoder(torch.nn.Module):
    def __init__(self, bottleneck, npts):
        super().__init__()
        self.tarch = get_arch(6, npts)
        self.N = int(np.prod([int(k) for k in self.tarch]))
        assert self.N == npts, "Number of tree outputs is %d, expected %d" % (self.N, npts)
        self.NFEAT = 8
        self.CODE_NFTS = bottleneck
        self.Nin = self.NFEAT + self.CODE_NFTS
        self.Nout = self.NFEAT
        self.N0 = int(self.tarch[0])
        self.nlevels = len(self.tarch)
        self.mlp = MLP([bottleneck, 256, 64, self.NFEAT * self.N0], bn=True)
        self.mlp_conv_list = torch.nn.ModuleList()
        bn = True
        for i in range(1, self.nlevels):
            if i == self.nlevels - 1:
                self.Nout = 3
                bn = False
            self.mlp_conv_list.append(CreateLevel(i, self.Nin, self.Nout, bn, self.tarch))

    def forward(self, code):
        level0 = self.mlp(code) #
        level0 = torch.tanh(level0)
        level0 = level0.view(-1, self.NFEAT, self.N0)  # (32, 8, 2)
        outs = [level0, ]
        for i in range(self.nlevels-1):
            inp = outs[-1]
            y = torch.unsqueeze(code, dim=2)  # (32, 1024, 1)
            y = y.repeat(1, 1, inp.shape[2])  # (32, 1024, 2)
            y = torch.cat([inp, y], dim=1)  # (32, 1032, 2)
            conv_outs = self.mlp_conv_list[i](y)
            outs.append(torch.tanh(conv_outs))
        return outs[-1].transpose(2, 1)
