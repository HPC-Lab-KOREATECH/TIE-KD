_base_ = [
    '../_base_/models/densedepth_resnet18.py', '../_base_/datasets/nyu_kd_with_b8_bts.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        type='DenseDepthHead_DPM_resnet',
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

