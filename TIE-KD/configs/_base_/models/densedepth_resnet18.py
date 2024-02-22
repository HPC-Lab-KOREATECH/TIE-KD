
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',

    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet18'),),

    decode_head=dict(
        type='DenseDepthHead_DPM_resnet',
        in_channels=[64, 64, 128, 256, 512],
        up_sample_channels=[128, 128, 256, 256, 512],
        channels=128, # last one
        align_corners=True, # for upsample
        loss_decode=dict(
            type='TIE_KD_loss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
