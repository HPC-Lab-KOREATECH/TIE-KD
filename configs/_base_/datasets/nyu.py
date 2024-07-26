# dataset settings
dataset_type = 'NYUDataset'
data_root = 'data/nyu/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop_Base', depth=True),
    dict(type='RandomRotate_Base', prob=0.5, degree=2.5),
    dict(type='RandomFlip_Base', prob=0.5),
    dict(type='RandomCrop_Base', crop_size=(416, 544)),
    dict(type='ColorAug_Base', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize_Base', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip_Base', direction='horizontal'),
            dict(type='Normalize_Base', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
        ])
]

# for visualization of pc
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip_Base', prob=0.0), # set to zero
    dict(type='Normalize_Base', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', 
         keys=['img'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_train.txt',
        pipeline=train_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10))

