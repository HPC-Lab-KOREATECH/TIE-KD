_base_ = [
    '../_base_/models/densedepth_mobilev2.py', '../_base_/datasets/nyu_kd_with_b8_depthformer.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    pretrained='mmcls://mobilenet_v2',
    decode_head=dict(
        type='DenseDepthHead54',
        conv_dim=49,
        n_bins=49,
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='TIE_KD_loss', loss_weight=0.1,
            n_bins_=50,
            sigma=0.15)
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

