

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
# from .ssim import SSIM
import math

# import time

@LOSSES.register_module()
class L_DPM_loss(nn.Module):
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
                 warm_iter=100,
                 n_bins_=257,
                 sigma=0.5):
        super(L_DPM_loss, self).__init__()

        self.loss_weight = loss_weight
        self.eps = 0.0000001 # avoid grad explode        

        self.num_zero_float=torch.tensor([0]).float().cuda() 
        self.num_min=torch.tensor([1e-16]).float().cuda() 
        self.num_1000_float=torch.tensor([1000]).float().cuda() 
        self.PI = torch.tensor(2.0*math.pi).cuda()
        self.sigma = torch.tensor([sigma]).cuda()  

        self.n_bins_ = n_bins_
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, depth_pred, kd_gt, bin_edges, out):    
        ##################
        B,_,H,W = kd_gt.shape        
        C = self.n_bins_-1        
        if depth_pred.shape[3] == 704: #KITTI   
            kd_gt = torchvision.transforms.functional.crop(kd_gt,110,0,H,W)
            depth_pred = torchvision.transforms.functional.crop(depth_pred,110,0,H,W)
            out = torchvision.transforms.functional.crop(out,110,0,H,W)        
            B,_,H,W = kd_gt.shape 
        ##################

        centers_tensor = 0.5 * (bin_edges[:-1] + bin_edges[1:]).contiguous().view(1,C, 1, 1)  
        centers_tensor = centers_tensor.repeat(B,1,H,W)        

        probability_T_kd = ( 1 / (torch.sqrt(self.PI * self.sigma**2)) * torch.exp(torch.negative(torch.pow(centers_tensor - kd_gt, 2) / (2 * torch.pow(self.sigma, 2)))))

        gaussian_softmax = torch.where(probability_T_kd<self.num_min,-self.num_1000_float,probability_T_kd)

        gaussian_softmax = self.softmax(gaussian_softmax)

        kl_loss_kd = torch.sum(torch.where(out!=self.num_zero_float,gaussian_softmax * torch.log(gaussian_softmax/(out+self.eps)+self.eps),self.num_zero_float),dim=1,keepdim=True)


        return kl_loss_kd 




