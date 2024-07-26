_base_ = [
    '../_base_/models/densedepth_mobilev2.py', '../_base_/datasets/nyu_kd_with_b8_depthformer.py',
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
            type='SI_loss', valid_mask=True, loss_weight=1.0)),
    )


find_unused_parameters=True

