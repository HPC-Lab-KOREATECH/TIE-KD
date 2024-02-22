# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from depth.models.builder import LOSSES
from .ssim import SSIM
import sys
@LOSSES.register_module()
class SSIM_loss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SSIM_loss, self).__init__()
        self.loss_weight = loss_weight

        self.eps = 0.001 # avoid grad explode
        self.SSIM_loss = SSIM()  


    def forward(self, depth_pred, kd_gt):
        """Forward function."""        

        SSIM_loss_value = self.loss_weight*(1-self.SSIM_loss(depth_pred, kd_gt))
        return SSIM_loss_value