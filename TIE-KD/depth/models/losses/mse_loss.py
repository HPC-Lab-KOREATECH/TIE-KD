# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from depth.models.builder import LOSSES
import torch.nn.functional as F
import sys
@LOSSES.register_module()
class MSE_loss(nn.Module):
    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(MSE_loss, self).__init__()
        self.loss_weight = loss_weight

    def loss_mse(self, input, target):
        loss = F.mse_loss(input,target)
        return loss

    def forward(self, depth_pred, kd_gt):
        """Forward function."""        
        loss_depth = self.loss_mse(depth_pred, kd_gt)

        return loss_depth