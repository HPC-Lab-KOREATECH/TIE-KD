_base_ = [
    '../_base_/models/densedepth_mobilev2.py', '../_base_/datasets/nyu_kd_with_b8_adabins.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    pretrained='mmcls://mobilenet_v2',
    decode_head=dict(
        type='DenseDepthHead_base_custom_kd',
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SSIM_loss', loss_weight=1.5)
        ),
    )

# max_lr=0.000357
find_unused_parameters=True






# _base_ = [
#     '../_base_/models/densedepth_mobile35.py', '../_base_/datasets/nyu_kd_with_b8.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
# ]

# model = dict(
#     pretrained='mmcls://mobilenet_v2',
#     decode_head=dict(
#         # type='DenseDepthHead54',
#         type='DenseDepthHead9',
#         conv_dim=65,
#         n_bins=65,
#         scale_up=True,
#         min_depth=1e-3,
#         max_depth=10,
#         loss_decode=dict(
#             type='SigLoss109', loss_weight=1.5,
#             n_bins_=66,
#             sigma=0.5)
#         ),
#     )

# # max_lr=0.000357
# find_unused_parameters=True

