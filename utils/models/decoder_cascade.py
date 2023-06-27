# @file decoder_cascade.py
# @author Junming Zhang, junming@umich.edu
# @brief cascade decoder
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

'''
Modified based on:

    https://github.com/paul007pl/VRCNet/blob/main/models/cascade.py

'''
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def symmetric_sample(points, num):
    '''
    Original implementation use FPS to sample points, but we use random sampling
    for simplicity.
    '''
    num_points = points.size(1)
    idx = np.random.choice(num_points, num)
    input_fps = points[:, idx, :]
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('fc_%d' % (i+1), nn.Linear(num_channels, dims[i+1]))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())
   
    def forward(self, features):
        return self.model(features)


class MLPConv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('conv1d_%d' % (i+1), nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())

    def forward(self, inputs):
        return self.model(inputs)


class ContractExpandOperation(nn.Module):
    def __init__(self, num_input_channels, up_ratio):
        super().__init__()
        self.up_ratio = up_ratio
        self.conv2d_1 = nn.Conv2d(num_input_channels, 64, kernel_size=(1, self.up_ratio), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):  # (32, 64, 2048)
        net = inputs.view(inputs.shape[0], inputs.shape[1], self.up_ratio, -1)  # (32, 64, 2, 1024)
        net = net.permute(0, 1, 3, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_1(net))  # (32, 64, 1024, 1)
        net = F.relu(self.conv2d_2(net))  # (32, 128, 1024, 1)
        net = net.permute(0, 2, 3, 1).contiguous()  # (32, 1024, 1, 128)
        net = net.view(net.shape[0], -1, self.up_ratio, 64)  # (32, 1024, 2, 64)
        net = net.permute(0, 3, 1, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_3(net)) # (32, 64, 1024, 2)
        net = net.view(net.shape[0], 64, -1)  # (32, 64, 2048)
        return net


class CascadeDecoder(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.coarse_mlp = MLP([bottleneck, 1024, 1024, 512 * 3])
        self.mean_fc = nn.Linear(1024, 128)
        self.up_branch_mlp_conv_mf = MLPConv([1157, 128, 64])
        self.up_branch_mlp_conv_nomf = MLPConv([bottleneck+5, 128, 64])
        self.contract_expand = ContractExpandOperation(64, 2)
        self.fine_mlp_conv = MLPConv([64, 512, 512, 3])

    def forward(self, code, inputs, step_ratio=2, num_extract=512, mean_feature=None):
        '''
        :param code: B * C
        :param inputs: B * C * N
        :param step_ratio: int
        :param num_extract: int
        :param mean_feature: B * C
        :return: coarse(B * N * C), fine(B, N, C)
        '''
        coarse = torch.tanh(self.coarse_mlp(code))  # (32, 1536)
        coarse = coarse.view(-1, 512, 3)  # (32, 512, 3)
        coarse = coarse.transpose(2, 1).contiguous()  # (32, 3, 512)

        inputs_new = inputs.view(-1, 2048, 3).contiguous()
        input_fps = symmetric_sample(inputs_new, int(num_extract/2))  # [32, 512,  3]
        input_fps = input_fps.transpose(2, 1).contiguous()  # [32, 3, 512]
        level0 = torch.cat([input_fps, coarse], 2)   # (32, 3, 1024)

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = gen_grid_up(2 ** (i + 1)).cuda().contiguous()
            grid = torch.unsqueeze(grid, 0)   # (1, 2, 2)
            grid_feat = grid.repeat(level0.shape[0], 1, 1024)   # (32, 2, 2048)
            point_feat = torch.unsqueeze(level0, 3).repeat(1, 1, 1, 2)  # (32, 3, 1024, 2)
            point_feat = point_feat.view(-1, 3, num_fine)  # (32, 3, 2048)
            global_feat = torch.unsqueeze(code, 2).repeat(1, 1, num_fine)  # (32, 1024, 2048)

            if mean_feature is not None:
                mean_feature_use = F.relu(self.mean_fc(mean_feature))  #(32, 128)
                mean_feature_use = torch.unsqueeze(mean_feature_use, 2).repeat(1, 1, num_fine)  #(32, 128, 2048)
                feat = torch.cat([grid_feat, point_feat, global_feat, mean_feature_use], dim=1)  # (32, 1157, 2048)
                feat1 = F.relu(self.up_branch_mlp_conv_mf(feat))  # (32, 64, 2048)
            else:
                feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)
                feat1 = F.relu(self.up_branch_mlp_conv_nomf(feat))  # (32, 64, 2048)

            feat2 = self.contract_expand(feat1) # (32, 64, 2048)
            feat = feat1 + feat2  # (32, 64, 2048)

            fine = self.fine_mlp_conv(feat) + point_feat  # (32, 3, 2048)
            level0 = fine

        return fine.transpose(1, 2), coarse.transpose(1, 2)


def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid


