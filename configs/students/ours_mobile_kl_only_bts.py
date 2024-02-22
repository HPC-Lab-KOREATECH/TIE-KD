_base_ = [
    '../_base_/models/densedepth_mobilev2.py', '../_base_/datasets/kitti_kd_with_b8_bts.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        type='DenseDepthHead_DPM',
        conv_dim=257,
        n_bins=257,
        scale_up=True,
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='L_DPM_loss', loss_weight=0.1,
            n_bins_=258,
            sigma=0.8)
        ),
    )

# max_lr=0.000357
find_unused_parameters=True

