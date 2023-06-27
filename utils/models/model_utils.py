# @file model_utils.py
# @author Junming Zhang, junming@umich.edu
# @brief ulitity functions for models
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import numpy as np
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear as Lin, Sequential as Seq, ReLU, BatchNorm1d
from torch_geometric.nn import fps, radius, PointConv


def MLP(channels, bn=True, last=False):
    if bn:
        l = [Seq(Lin(channels[i - 1], channels[i], bias=False), BatchNorm1d(channels[i]), ReLU())
                for i in range(1, len(channels)-1)]
    else:
        l = [Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())
                for i in range(1, len(channels)-1)]

    if last:
        l.append(Seq(Lin(channels[-2], channels[-1], bias=True)))
    else:
        if bn:
            l.append(Seq(Lin(channels[-2], channels[-1], bias=False), BatchNorm1d(channels[-1]), ReLU()))
            # l.append(Seq(Lin(channels[-2], channels[-1], bias=False), BatchNorm1d(channels[-1])))
        else:
            l.append(Seq(Lin(channels[-2], channels[-1], bias=True), ReLU()))
            
    return Seq(*l)