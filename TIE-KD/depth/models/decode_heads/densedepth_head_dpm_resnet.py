from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding
from torch.nn.modules import conv

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
from depth.models.utils import UpConvBlock, BasicConvBlock

from depth.ops import resize
import sys
from depth.models.builder import build_loss
import matplotlib.pyplot as plt
import numpy as np

class UpSample_custom(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample_custom, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        up_x = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x))

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))

@HEADS.register_module()
class DenseDepthHead_DPM_resnet(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
        fpn (bool): Whether apply FPN head.
            Default: False
        conv_dim (int): Default channel of features in FPN head.
            Default: 256.
    """

    def __init__(self,
                 up_sample_channels,
                 fpn=False,
                 conv_dim=256,
                 **kwargs):
        super(DenseDepthHead_DPM_resnet, self).__init__(**kwargs)

        self.conv_out = nn.Sequential(nn.Conv2d(128, self.n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = self.in_channels[::-1]

        self.conv_list = nn.ModuleList()
        up_channel_temp = 0

        self.upsample_custom = UpSample_custom(skip_input=128,
                                 output_features=128,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        self.upsample_custom2 = UpSample_custom(skip_input=128,
                                 output_features=128,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)

        self.fpn = fpn
        if self.fpn:
            self.num_fpn_levels = len(self.in_channels)

            # construct the FPN
            self.lateral_convs = nn.ModuleList()
            self.output_convs = nn.ModuleList()

            for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
                lateral_conv = ConvModule(
                    in_channel, conv_dim, kernel_size=1, norm_cfg=self.norm_cfg
                )
                output_conv = ConvModule(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        else:
            for index, (in_channel, up_channel) in enumerate(
                    zip(self.in_channels, self.up_sample_channels)):
                if index == 0:
                    self.conv_list.append(
                        ConvModule(
                            in_channels=in_channel,
                            out_channels=up_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act_cfg=None
                        ))
                else:
                    self.conv_list.append(
                        UpSample(skip_input=in_channel + up_channel_temp,
                                 output_features=up_channel,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg))

                # save earlier fusion target
                up_channel_temp = up_channel
        
        # self.conv_out = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
        #                               nn.Softmax(dim=1)) #
    def forward(self, inputs, img_metas):
        """Forward function."""
        temp_feat_list = []
        if self.fpn:            
            for index, feat in enumerate(inputs[::-1]):
                x = feat
                lateral_conv = self.lateral_convs[index]
                output_conv = self.output_convs[index]
                cur_fpn = lateral_conv(x)

                # Following FPN implementation, we use nearest upsampling here. Change align corners to True.
                if index != 0:
                    y = cur_fpn + F.interpolate(temp_feat_list[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
                else:
                    y = cur_fpn
                    
                y = output_conv(y)
                temp_feat_list.append(y)

        else:
            temp_feat_list = []
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat = self.conv_list[index](up_feat, skip_feat)
                    temp_feat_list.append(temp_feat)

        ######################### 
        up_feat = self.upsample_custom(temp_feat_list[-1])
        out = self.conv_out(up_feat)


        B,C,W,H = out.shape
        bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins+1).cuda()


        centers = 0.5 * (bins[:-1] + bins[1:]) #center
        centers=centers.unsqueeze(dim=0)
        centers = centers.repeat(B,1)
        centers = centers.contiguous().view(B, C, 1, 1)  
     
        output = torch.sum(out*centers,dim=1,keepdim=True)

        return output, bins, out
        
    
    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg,kd_gt):
        depth_pred, bin_edges, out= self.forward(inputs, img_metas)

        losses = dict()

        # losses["loss_depth"] = self.loss_decode(depth_pred, depth_gt, kd_gt, bin_edges, out)
        losses["loss_depth"] = self.loss_decode(depth_pred, kd_gt, bin_edges, out)

       
        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):

        depth_pred, bin_edges, out = self.forward(inputs, img_metas)
        
        return depth_pred
    