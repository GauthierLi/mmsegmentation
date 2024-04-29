_base_ = [
    '../_base_/models/san_vit-b16.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# By default, models are trained on 4 GPUs with 8 images per GPU
train_dataloader = dict(batch_size=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  # noqa
# model = dict(
#     pretrained=pretrained,
#     text_encoder=dict(dataset_name='loveda'),
#     decode_head=dict(num_classes=6))

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=640,
    test_cfg=dict(size_divisor=32))

num_classes = 6
model = dict(
    _delete_=True,
    type='MultimodalEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    asymetric_input=True,
    encoder_resolution=0.5,
    image_encoder=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=3,
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=True,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        act_cfg=dict(type='QuickGELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        frozen_exclude=['pos_embed']),
    text_encoder=dict(
        type='CLIPTextEncoder',
        dataset_name='loveda',
        templates='vild',
        embed_dims=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4,
        output_dims=512,
        cache_feature=True,
        cat_bg=True,
        norm_cfg=dict(type='LN', eps=1e-5)
        ),
    decode_head=dict(
        type='SideAdapterCLIPHead',
        num_classes=num_classes,
        deep_supervision_idxs=[7],
        san_cfg=dict(
            in_channels=3,
            clip_channels=768,
            embed_dims=240,
            patch_size=16,
            patch_bias=True,
            num_queries=100,
            cfg_encoder=dict(
                num_encode_layer=8,
                num_heads=6,
                mlp_ratio=4
            ),
            fusion_index=[0, 1, 2, 3],
            cfg_decoder=dict(
                num_heads=12,
                num_layers=1,
                embed_channels=256,
                mlp_channels=256,
                num_mlp=3,
                rescale=True),
            norm_cfg=dict(type='LN', eps=1e-6),
        ),
        maskgen_cfg=dict(
            sos_token_format='cls_token',
            sos_token_num=100,
            cross_attn=False,
            num_layers=3,
            embed_dims=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            out_dims=512,
            final_norm=True,
            act_cfg=dict(type='QuickGELU'),
            norm_cfg=dict(type='LN', eps=1e-5),
            frozen_exclude=[]
        ),
        align_corners=False,
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ])),
        loss_decode=[dict(type='CrossEntropyLoss',
                          loss_name='loss_cls_ce',
                          loss_weight=2.0,
                          class_weight=[1.0] * num_classes + [0.1]),
                     dict(type='CrossEntropyLoss',
                          use_sigmoid=True,
                          loss_name='loss_mask_ce',
                          loss_weight=5.0),
                     dict(type='DiceLoss',
                          ignore_index=None,
                          naive_dice=True,
                          eps=1,
                          loss_name='loss_mask_dice',
                          loss_weight=5.0)
                     ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable

max_iters = 368750
# training schedule for 60k
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=500,
    val_begin=55000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU'))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(max_iters * 0.88), int(max_iters * 0.95)],
    gamma=0.1)
