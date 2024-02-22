

_base_ = [
    '../_base_/models/densedepth_mobilev2.py', '../_base_/datasets/kitti_kd_with_b8_depthformer.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        type='DenseDepthHead_base_custom_kd',
        scale_up=True,
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SI_loss')
        ),
    )

# max_lr=0.000357
find_unused_parameters=True
