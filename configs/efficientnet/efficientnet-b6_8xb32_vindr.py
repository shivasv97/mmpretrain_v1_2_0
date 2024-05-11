_base_ = [
    '../_base_/models/efficientnet_b6.py',
    '../_base_/datasets/medical_datasets/vindr_cxr_cls.py',
    '../_base_/schedules/medical_schedules/vindr_cxr_cls_sched.py',
    '../_base_/default_runtime.py',
]

load_from='/scratch/ssenth21/bench_boost_lclz/mmpretrain_v1_2_0/chkpts/efficientnet-b6_3rdparty-ra-noisystudent_in1k_20221103-7de7d2cc.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b6'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=6,
        in_channels=2304,
        #loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        topk=None,
    ))

