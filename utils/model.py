# @file model.py
# @author Junming Zhang, junming@umich.edu; Haomeng Zhang, haomeng@umich.edu
# @brief Model class
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import torch.nn.functional as F
from .models.decoder_folding import FoldingBasedDecoder
from .models.decoder_topnet import TopNetDecoder
from .models.decoder_cascade import CascadeDecoder
from .models.decoder_PCN import PCNDecoder, PCNEncoder
from .models.decoder_snowflakenet import SnowFlakeNetDecoder, \
    compute_snowflakenet_decoder_loss, HyperSnowFlakeNetDecoder
from .models.encoders import PointNetEncoder, PointNet2Encoder, Classifier, \
    HypersphericalModule, Segmentator, PointNet2EncoderHyper
from .models.encoder_dgcnn import DGCNNEncoder
from .models.encoder_snowflakenet import SnowFlakeNetEncoder, HyperSnowFlakeNetEncoder
from .ChamferDistancePytorch.chamfer3D import dist_chamfer_3D


class Model(torch.nn.Module):
    """
    Arguments:
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # define encoder
        if self.args.encoder_choice == 'pointnet':
            self.encoder = PointNetEncoder([3, 64, 256, 512, self.args.maxpool_bottleneck], bn=True)
        elif self.args.encoder_choice == 'pointnet2':
            self.encoder = PointNet2Encoder(self.args.maxpool_bottleneck)
        elif self.args.encoder_choice == 'hpointnet2':
            self.encoder = PointNet2EncoderHyper(self.args.maxpool_bottleneck,
                    args.use_hyperspherical_module, args.use_hyperspherical_encoding)
        elif self.args.encoder_choice == 'pcn':
            self.encoder = PCNEncoder(self.args.maxpool_bottleneck)
        elif self.args.encoder_choice == 'dgcnn':
            self.encoder = DGCNNEncoder(self.args.maxpool_bottleneck)
        elif self.args.encoder_choice == 'snowflake':
            self.encoder = SnowFlakeNetEncoder(self.args.maxpool_bottleneck)
        elif self.args.encoder_choice == 'hsnowflake':
            self.encoder = HyperSnowFlakeNetEncoder(self.args.maxpool_bottleneck)
        else:
            raise ValueError('{} encoder has not been implemented yet.'.format(
                self.args.encoder_choice))

        # define hyperspherical module
        in_channels = self.args.maxpool_bottleneck
        if self.args.use_hyperspherical_module:
            self.hyperspherical_module = HypersphericalModule(
                [self.args.maxpool_bottleneck] + [self.args.hyper_bottleneck for _ in range(self.args.hyperspherical_module_layers)],
                self.args.use_hyperspherical_encoding, self.args.norm_order, self.args.hyperspherical_module_BN)

        # define decoder
        if 'classification' in self.args.task:
            mlp_dims = [in_channels]
            mlp_dims += [int(dim) for dim in self.args.mlps_classifier.split(',')]
            self.decoder_classification = Classifier(mlp_dims, bn=self.args.use_BN_classifier)

        if 'segmentation' in self.args.task:
            mlp_dims = [in_channels+3]
            mlp_dims += [int(dim) for dim in self.args.mlps_segmentator.split(',')]
            self.decoder_segmentation = Segmentator(mlp_dims, bn=self.args.use_BN_segmentator)

        if 'completion' in self.args.task:
            if self.args.completion_decoder_choice == 'folding':
                self.decoder_completion = FoldingBasedDecoder(in_channels)
            elif self.args.completion_decoder_choice == 'topnet':
                self.decoder_completion = TopNetDecoder(in_channels, 2048)
            elif self.args.completion_decoder_choice == 'cascade':
                self.decoder_completion = CascadeDecoder(in_channels)
            elif self.args.completion_decoder_choice == 'pcn':
                self.decoder_completion = PCNDecoder(in_channels)
            elif self.args.completion_decoder_choice == 'snowflake':
                self.decoder_completion = SnowFlakeNetDecoder(in_channels)
            elif self.args.completion_decoder_choice == 'hsnowflake':
                self.decoder_completion = HyperSnowFlakeNetDecoder(in_channels,
                        self.args.use_hyperspherical_module, self.args.use_hyperspherical_encoding)
            else:
                raise ValueError('{} decoder has not been supported yet.'.format(
                    self.args.completion_decoder_choice))

        
        # Add weights for uncertainty
        if self.args.uncertainty_flag:
            self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.w1.data.fill_(1)
            self.w2.data.fill_(1)
        

    def forward(self, x, pos, batch):
        batch = batch - batch.min()
        x = self.encoder(None, pos, batch)
        encoding_feature = x
        self.encoding_feature = x

        # hyperspherical module 
        if self.args.use_hyperspherical_module:
            x = self.hyperspherical_module(x)
            self.encoding_feature = x[0]
            if self.args.use_hyperspherical_encoding:
                encoding_feature = x[1]
            else:
                encoding_feature = x[0]


        # classification branch
        if 'classification' in self.args.task:
            self.output_cls_logits = self.decoder_classification(encoding_feature)
            self.pred_classification = self.output_cls_logits.max(1)[1]

        # segmentation branch
        if 'segmentation' in self.args.task:
            self.output_seg_logits = self.decoder_segmentation(
                encoding_feature, pos, batch)
            self.pred_segmentation = self.output_seg_logits.max(1)[1]

        # point cloud completion branch
        if 'completion' in self.args.task:
            if self.args.completion_decoder_choice == 'cascade':
                self.pred_completion, self.coarse_completion = self.decoder_completion(encoding_feature, pos)
            elif self.args.completion_decoder_choice == 'pcn':
                self.pred_completion, self.coarse_completion = self.decoder_completion(encoding_feature)
            elif self.args.completion_decoder_choice in ['snowflake', 'hsnowflake']:
                self.pred_completion_list = self.decoder_completion(encoding_feature, pos)
                self.pred_completion = self.pred_completion_list[-1]
                self.pos = pos
            else:
                self.pred_completion = self.decoder_completion(encoding_feature)

    def compute_loss(self, category, pc_label, w1=1.0, w2=1.0):
        loss = 0.0
        if 'classification' in self.args.task:
            loss_classification = F.nll_loss(self.output_cls_logits, category, reduction='none')
            self.loss_classification = loss_classification
            loss += w1 * loss_classification.mean()
        if 'segmentation' in self.args.task:
            # category here is point-wise category ground truth
            loss_segmentation = F.nll_loss(self.output_seg_logits, category, reduction='none')
            self.loss_segmentation = loss_segmentation
            loss += loss_segmentation.mean()
        if 'completion' in self.args.task:
            loss_completion = self.compute_chamfer_distance(self.pred_completion,
                    pc_label.view(self.pred_completion.size(0), -1, 3))
            self.loss_completion = loss_completion.clone()
            if self.args.completion_decoder_choice in ['pcn', 'cascade']:
                loss_completion += 0.1*self.compute_chamfer_distance(self.coarse_completion,
                        pc_label.view(self.pred_completion.size(0), -1, 3))
            elif self.args.completion_decoder_choice in ['snowflake', 'hsnowflake']:
                loss_completion = compute_snowflakenet_decoder_loss(
                        self.pred_completion_list,
                        self.pos.view(self.pred_completion.size(0), -1, 3),
                        pc_label.view(self.pred_completion.size(0), -1, 3),
                        self.training)
            loss += w2 * loss_completion.mean()


        # SEC loss
        # Spherical embedding constraint, which constrains the embeddings to
        # lie on the surface of the same hypersphere before noramlization
        self.loss_sec = None
        if self.args.use_hyperspherical_module \
                and self.args.use_hyperspherical_encoding \
                and self.args.weight_sec_loss is not None:
            encoding_norm = self.encoding_feature.norm(dim=-1)
            mu = encoding_norm.mean()
            loss_sec = (encoding_norm - mu)**2
            self.loss_sec = loss_sec
            loss = loss + self.args.weight_sec_loss*loss_sec.mean()


        if self.args.uncertainty_flag:
            loss = 1 / (torch.square(w1)) * loss_classification + 1 / (2 * torch.square(w2)) * loss_completion + torch.log(torch.square(w1)) + torch.log(torch.square(w2))
        self.loss = loss
        return loss


    def compute_chamfer_distance(self, pred, gt, calc_f1=False):
        cham_loss = dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, _, _ = cham_loss(gt, pred)
        if self.training:
            dist = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
        else:
            # During eval, make consistent with metrics used in the benchmark of
            # Completion3D
            dist = (dist1.mean(1) + dist2.mean(1))
        return dist

