
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        type='MobileNetV2_base',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),

    decode_head=dict(
        type='DenseDepthHead_base_custom_kd',
        in_channels=[24,32,96,320],
        up_sample_channels=[128, 256, 512, 2048],
        channels=128, # last one
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SI_loss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
