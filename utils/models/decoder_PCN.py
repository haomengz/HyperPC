# @file decoder_PCN.py
# @author Junming Zhang, junming@umich.edu
# @brief PCN decoder
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import numpy as np
import torch.nn.functional as F


class PCNEncoder(torch.nn.Module):
    def __init__(self, embed_size, bn=True):
        super().__init__()
        bias = False if bn else True
        self.bn = bn
        self.conv1 = torch.nn.Conv1d(3, 128, kernel_size=1, bias=bias)
        self.conv2 = torch.nn.Conv1d(128, 256, kernel_size=1, bias=bias)
        self.conv3 = torch.nn.Conv1d(512, 512, kernel_size=1, bias=bias)
        self.conv4 = torch.nn.Conv1d(512, embed_size, kernel_size=1, bias=bias)
        self.relu = torch.nn.ReLU()

        if bn:
            self.bn1 = torch.nn.BatchNorm1d(128)
            self.bn2 = torch.nn.BatchNorm1d(256)
            self.bn3 = torch.nn.BatchNorm1d(512)
            self.bn4 = torch.nn.BatchNorm1d(embed_size)

    def forward(self, x, pos, batch):
        '''
        :pos: B*N * 3
        :return: B * C
        '''
        batch_size = batch.max() + 1
        inputs = pos.view(batch_size, -1, 3).transpose(1, 2)

        if self.bn:
            features = self.relu(self.bn1(self.conv1(inputs)))
            features = self.relu(self.bn2(self.conv2(features)))
        else:
            features = self.relu(self.conv1(inputs))
            features = self.relu(self.conv2(features))

        features_global, _ = torch.max(features, 2, keepdim=True)  # [32, 256, 1]
        features_global_tiled = features_global.repeat(1, 1, inputs.shape[2])  # [32, 256, 2048]
        features = torch.cat([features, features_global_tiled], dim=1)  # [32, 512, 2048]

        if self.bn:
            features = self.relu(self.bn3(self.conv3(features)))
            features = self.relu(self.bn4(self.conv4(features)))
        else:
            features = self.relu(self.conv3(features))
            features = self.relu(self.conv4(features))

        features, _ = torch.max(features, 2)  # [32, embed_size]
        return features


class PCNDecoder(torch.nn.Module):
    def __init__(self, bottleneck):
        super(PCNDecoder, self).__init__()
        self.num_coarse = 512
        self.num_fine = 2048
        self.grid_size = 2
        self.fc1 = torch.nn.Linear(bottleneck, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, self.num_coarse * 3)

        self.conv1 = torch.nn.Conv1d(3+2+bottleneck, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)


    def forward(self, features):
        bsize = features.size()[0]
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        coarse = self.fc3(x).view(-1, 3, self.num_coarse)

        grid = torch.meshgrid(torch.linspace(-0.05, 0.05, self.grid_size), torch.linspace(-0.05, 0.05, self.grid_size))
        grid = torch.unsqueeze(torch.stack(grid, dim=2).view(-1, 2), 0)
        grid_feat = grid.repeat([features.size(0), self.num_coarse, 1]).transpose(1, 2).cuda()

        point_feat = coarse.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.grid_size**2, 1)
        point_feat = point_feat.view(-1, self.num_fine, 3).transpose(1, 2).contiguous()

        global_feat = features.unsqueeze(2).repeat(1, 1, self.num_fine)
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)

        fine = F.relu(self.conv1(feat))
        fine = F.relu(self.conv2(fine))
        fine = self.conv3(fine) + point_feat
        return fine.transpose(1, 2), coarse.transpose(1, 2)
