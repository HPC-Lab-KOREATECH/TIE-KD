

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
from .l_depth_loss import L_DEPTH_loss
from .l_dpm_loss import L_DPM_loss
import math

# import time

@LOSSES.register_module()
class TIE_KD_loss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq133/AdaBins/blob/main/loss.py>`_.

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
                 warm_iter=100,
                 n_bins_=257,
                 sigma=0.5):
        super(TIE_KD_loss, self).__init__()
        self.loss_weight = loss_weight

        # self.valid_mask = valid_mask        
        # self.max_depth = max_depth
        # self.eps = 0.0000001 # avoid grad explode        

        # self.num_zero_float=torch.tensor([0]).float().cuda() 
        # self.num_min=torch.tensor([1e-16]).float().cuda() 
        # self.num_1000_float=torch.tensor([1000]).float().cuda() 
        # self.PI = torch.tensor(math.pi).cuda()
        # self.sigma = torch.tensor([sigma]).cuda()  

        # self.n_bins_ = n_bins_
        # self.SSIM_loss = SSIM()           
        # self.softmax = nn.Softmax(dim=1)

        self.L_DEPTH = L_DEPTH_loss(n_bins_=n_bins_)
        self.L_DPM = L_DPM_loss(n_bins_=n_bins_)


    

    def forward(self, depth_pred, kd_gt, bin_edges, out):    
        
        L_DEPTH_value = self.L_DEPTH(depth_pred, kd_gt, bin_edges, out)
        L_DPM_value = self.L_DPM(depth_pred, kd_gt, bin_edges, out)


        return 10*(self.loss_weight*torch.mean(L_DPM_value) + (1-self.loss_weight)*L_DEPTH_value)




