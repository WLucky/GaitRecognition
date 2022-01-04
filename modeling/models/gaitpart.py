import torch
import torch.nn as nn
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs
from ..modules import BasicConv2d, FocalConv2d
from util_tools import clones
import pdb


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max


    def forward(self, x):
        """
          Input:  x,   [n, s, c, p]
          Output: ret, [n, p, c]
        """
        n, s, c, p = x.size()
        x = x.permute(3, 0, 2, 1).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 0, 2).contiguous()  # [n, p, c]
        return ret


class GaitPart(nn.Module):
    def __init__(self, *args, **kargs):
        super(GaitPart, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """
        self.Backbone = self.get_backbone()
        self.Head = SeparateFCs(parts_num=16, in_channels=128, out_channels=128)
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=[16]))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=128, parts_num=16))
        
        self.init_parameters()

    def get_backbone(self):
        layers = [BasicConv2d(1, 32, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(inplace=True)]
        layers += [BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [FocalConv2d(32, 64, kernel_size=3, stride=1, padding=1, halving=2), nn.LeakyReLU(inplace=True)]
        layers += [FocalConv2d(64, 64, kernel_size=3, stride=1, padding=1, halving=2), nn.LeakyReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [FocalConv2d(64, 128, kernel_size=3, stride=1, padding=1, halving=3), nn.LeakyReLU(inplace=True)]
        layers += [FocalConv2d(128, 128, kernel_size=3, stride=1, padding=1, halving=3), nn.LeakyReLU(inplace=True)]

        return nn.Sequential(*layers)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
     
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  #torch.Size([32, 30, 64, 44])
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2) #torch.Size([32, 30, 1, 64, 44])

        del ipts
        out = self.Backbone(sils)  # [n, s, c, h, w]  torch.Size([32, 30, 128, 16, 11])
        out = self.HPP(out)  # [n, s, c, p]  torch.Size([32, 30, 128, 16])
        out = self.TFA(out, seqL)  # [n, p, c] torch.Size([32, 16, 128])
        embs = self.Head(out.permute(1, 0, 2).contiguous())  # [p, n, c]
        embs = embs.permute(1, 0, 2).contiguous()  # [n, p, c] torch.Size([32, 16, 128])

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval

def gaitPart():
    return GaitPart()
