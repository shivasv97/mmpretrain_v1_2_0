# dataset settings
dataset_type = 'MultiLabelDataset'
data_preprocessor = dict(
    num_classes=6,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

metainfo = {
    'classes': ["Pleural effusion", "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", "No finding"]
    }

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=1024),
    dict(type='Albu', transforms=[
    dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2)]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
                type='Resize',
                scale=(1024, 1024),
                interpolation='bicubic',
                backend='pillow'),
    #dict(type='ResizeEdge', scale=1024, edge='short'),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/',
        ann_file='annotations/image_labels_train.json',
        metainfo=metainfo,
        data_prefix='train_jpeg/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/',
        ann_file='annotations/image_labels_test.json',
        metainfo=metainfo,
        data_prefix='test_jpeg/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='AveragePrecision')#, topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
