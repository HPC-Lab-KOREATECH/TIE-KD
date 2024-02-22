

import torch
import torch.nn as nn

from depth.models.builder import LOSSES

import torch.nn.functional as F
from depth.ops import resize
from PIL import Image
# import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision
from .ssim import SSIM
import math

# import time

@LOSSES.register_module()
class L_DEPTH_loss(nn.Module):
    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100,
                 n_bins_=257,
                 sigma=0.5):
        super(L_DEPTH_loss, self).__init__()

        self.loss_weight = loss_weight
        self.SSIM_loss = SSIM()   

    def forward(self, depth_pred, kd_gt, bin_edges, out):    
        ##################
        B,_,H,W = kd_gt.shape        
        kd_gt = torchvision.transforms.functional.crop(kd_gt,110,0,H,W)
        depth_pred = torchvision.transforms.functional.crop(depth_pred,110,0,H,W)
        out = torchvision.transforms.functional.crop(out,110,0,H,W)
        ##################
        
        SSIM_loss_value = (1-self.SSIM_loss(depth_pred, kd_gt))
        return SSIM_loss_value




