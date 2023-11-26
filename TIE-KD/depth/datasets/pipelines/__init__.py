# Copyright (c) OpenMMLab. All rights reserved.
from .compose import Compose
from .formating import (Collect, ImageToTensor, ToDataContainer, ToTensor,
                        Transpose, to_tensor)
from .test_time_aug import MultiScaleFlipAug

from .loading import DepthLoadAnnotations, DisparityLoadAnnotations, LoadImageFromFile, LoadKITTICamIntrinsic, KDLoadAnnotations, KDLoadAnnotations_NYU
# from .transforms_kd import KBCrop_kd, RandomRotate_kd, RandomFlip_kd, RandomCrop_kd, NYUCrop_kd, Resize_kd, Normalize_kd

# from .transforms_base import KBCrop, RandomRotate, RandomFlip, RandomCrop, NYUCrop, Resize, Normalize
# from .transforms_kd import KBCrop, RandomRotate, RandomFlip, RandomCrop, NYUCrop, Resize, Normalize
from .transforms_kd2 import KBCrop, RandomRotate, RandomFlip, RandomCrop, NYUCrop, Resize, Normalize



from .formating import DefaultFormatBundle

__all__ = [
    'Compose', 'Collect', 'ImageToTensor', 'ToDataContainer', 'ToTensor',
    'Transpose', 'to_tensor', 'MultiScaleFlipAug',

    'DepthLoadAnnotations', 'KBCrop', 'RandomRotate', 'RandomFlip', 'RandomCrop', 'DefaultFormatBundle',
    'NYUCrop', 'DisparityLoadAnnotations', 'Resize', 'LoadImageFromFile', 'Normalize', 'LoadKITTICamIntrinsic', 'KDLoadAnnotations'
    # 'KBCrop_kd', 'RandomRotate_kd', 'RandomFlip_kd', 'RandomCrop_kd', 'NYUCrop_kd', 'Resize_kd', 'Normalize_kd'
]