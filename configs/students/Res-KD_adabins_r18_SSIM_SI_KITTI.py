

_base_ = [
    '../_base_/models/densedepth_resnet18.py', '../_base_/datasets/kitti_kd_with_b8_adabins.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        type='DenseDepthHead_base_custom_kd_resnet',
        scale_up=True,
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SSIM_SI_loss', loss_weight=1.5)
        ),
    )

# max_lr=0.000357
find_unused_parameters=True
