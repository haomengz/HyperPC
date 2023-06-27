# @file encoders.py
# @author Junming Zhang, junming@umich.edu; Haomeng Zhang, haomeng@umich.edu
# @brief encoder for piontnet, pointnet++
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, knn, global_max_pool, DynamicEdgeConv
from .model_utils import MLP


class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_var_cls = torch.nn.Parameter(torch.zeros(1))
        self.log_var_rec = torch.nn.Parameter(torch.zeros(1))

    def forward(self, loss_cls, loss_rec):
        loss_cls = loss_cls * torch.exp(-self.log_var_cls) + self.log_var_cls
        loss_rec = loss_rec * torch.exp(-self.log_var_rec) + self.log_var_rec
        return loss_cls + loss_rec


class Classifier(torch.nn.Module):
    """
    Classifier to do classification from the latent feature vector
    """
    def __init__(self, channels, bn=True):
        super(Classifier, self).__init__()
        self.mlp = MLP(channels, bn=bn, last=True)

    def forward(self, x):
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)


class HypersphericalModule(torch.nn.Module):
    def __init__(self, channels, is_normalized, norm_order, is_BN):
        '''
        Transform encoding features (output from max pooling layer) to features
        with hyper_bneck dimensions, and the output will be normalized (L2) if
        the flag is_normalized is set to True
        '''
        super().__init__()
        self.is_BN = is_BN
        if is_BN:
            self.bn = torch.nn.BatchNorm1d(channels[0])
        self.mlp = MLP(channels, bn=False, last=True)
        self.is_normalized = is_normalized
        self.norm_order = norm_order

    def forward(self, x):
        '''
        x: [b, f]
        '''
        if self.is_BN:
            x = self.mlp(F.relu(self.bn(x)))
        else:
            x = self.mlp(x)
        out = [x]
        if self.is_normalized:
            out.append(x / x.norm(dim=-1, keepdim=True, p=self.norm_order))
        return out


class PointNetEncoder(torch.nn.Module):
    def __init__(self, mlp_dims, bn=True):
        super(PointNetEncoder, self).__init__()
        self.mlp = MLP(mlp_dims, bn=bn, last=False)

    def forward(self, x, pos, batch):
        bsize = batch.max() + 1
        x = self.mlp(pos)
        encoding = global_max_pool(x, batch)
        return encoding


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)
        # self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        # x = self.nn(pos)
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2EncoderHyper(torch.nn.Module):
    '''
    The pointnet++ encoder. Each pointnet structure will output hyper enbedding
    '''
    def __init__(self, bottleneck, use_hyper_module, use_hyper_encoding):
        super().__init__()

        self.use_hyper_module = use_hyper_module
        self.use_hyper_encoding = use_hyper_encoding

        self.sa1_module = SAModule(0.25, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, bottleneck]))

        self.mlp1, self.mlp2 = None, None
        if self.use_hyper_module:
            self.mlp1 = MLP([128, 128], bn=False, last=True)
            self.mlp2 = MLP([256, 256], bn=False, last=True)

    def forward(self, x, pos, batch):
        l1_x, l1_pos, l1_batch = self.sa1_module(x, pos, batch)
        l1_x = self.hyper(l1_x, self.mlp1)

        l2_x, l2_pos, l2_batch = self.sa2_module(l1_x, l1_pos, l1_batch)
        l2_x = self.hyper(l2_x, self.mlp2)

        l3_x, l3_pos, l3_batch = self.sa3_module(l2_x, l2_pos, l2_batch)

        # This is for some models that need hierarchicial information
        bsize = batch.max() + 1
        self.l_points = {
            'l0_xyz': self.reshape(pos, bsize),
            'l0_points': self.reshape(pos, bsize),
            'l1_xyz': self.reshape(l1_pos, bsize),
            'l1_points': self.reshape(l1_x, bsize),
            'l2_xyz': self.reshape(l2_pos, bsize),
            'l2_points': self.reshape(l2_x, bsize),
            'l3_xyz': self.reshape(l3_pos, bsize)}
        return l3_x

    def hyper(self, x, mlp=None):
        if self.use_hyper_module:
            x = mlp(x)
            if self.use_hyper_encoding:
                x = x.div(x.norm(dim=-1, keepdim=True))
        return x

    def reshape(self, x, bsize):
        '''
        Reshape the flatten inputs

        x: [B*N, f]
        return: [B, f, N]
        '''
        f = x.shape[-1]
        x = x.view(bsize, -1, f).transpose(1, 2).contiguous()
        return x


class PointNet2Encoder(torch.nn.Module):
    '''
    The pointnet++ encoder 
    '''
    def __init__(self, bottleneck):
        super().__init__()

        self.sa1_module = SAModule(0.25, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, bottleneck]))

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return x


class Segmentator(torch.nn.Module):
    """
    Pointwise part segmentation

    Arguments:
        k: the number of output categories. For ShapeNet, k is 50. 
    """
    def __init__(self, channels, bn=True):
        super(Segmentator, self).__init__()
        self.mlp = MLP(channels, bn=bn, last=True)

    def forward(self, encoding, pos, batch):
        '''
        encoding: encoding feature for point clouds
        pos: xyz
        batch: batch indices for xyz
        '''
        encoding_pointwise = encoding[batch]
        x = torch.cat([encoding_pointwise, pos], dim=-1)
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)
