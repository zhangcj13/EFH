img_h = 256 
img_w = 352
ori_img_h = 260
ori_img_w = 346
multiscale_range = 5

num_classes = 3

batch_size = 64 #128

net = dict(
    type='MultiHeadsDetector',
)
arch_net = {0:'resnet10', 
            1:'resnet18',
            2:'resnet34',
            3:'resnet50',
            11:'resnet18t',
            21:'resnet34t'}[3]

featuremap_out_channel = [64, 128, 256, 512] if arch_net not in ['resnet10','resnet18t','resnet34t'] else [64, 64, 128, 256]

backbone = dict(
    type='ResNetWrapper',
    resnet=arch_net, # resnet18
    pretrained=True,
    in_channels=featuremap_out_channel,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

width  = {'resnet10' :0.25,
          'resnet18t':0.25,
          'resnet18' :0.5,
          'resnet34t':0.25,
          'resnet34' :0.5,
          'resnet50' :2.0}[arch_net]

neck = dict(
    type='PAFPN',
    in_channels=[256, 512, 1024],
    width=width,
    depth=0.33,
    depthwise=False,
    act="silu",
)

# data for box detect
max_boxes = 256

bbox_heads = dict(type='YOLOX_Head',
                num_classes=num_classes,
                width=width,
                strides=[8, 16, 32],
                in_channels=[256, 512, 1024],
                act="silu",
                depthwise=False)

no_aug_epochs = 20
epochs = 300 #100
num_training_sample= 47998 #1712
num_iters_per_epoch = num_training_sample // batch_size
total_iter = num_iters_per_epoch * epochs

# training param
optimizer={'SGD'    :dict(type='SGD', lr = 0.01/64 * batch_size, momentum=0.9,nesterov=True),
           'RMSprop':dict(type='RMSprop', lr=0.00002, weight_decay=1e-8, momentum=0.9),
           'AdamW'  :dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), eps=1e-8),
           'Adam'   :dict(type='AdamW', lr=1e-3, betas = (0.937, 0.999)),
}['SGD']
paramgroup=['weight_bias_bn'][0]

scheduler = {'PolyLR'     : dict(type = 'PolyLR', max_iters=total_iter, power=0.9),
             'StepLR'     : dict(type = 'StepLR', step_size=1, gamma=0.95),
             'MultiStepLR': dict(type = 'MultiStepLR', milestones=[30, 70, 100], gamma=0.2),
             'CAWR'       : dict(type = 'CosineAnnealingWarmRestarts', T_0=10, T_mult=2, eta_min=1),
             'CAL'        : dict(type = 'CosineAnnealingLR', T_max=5, eta_min=0, last_epoch=-1, verbose=False),
             'YWC'        : dict(type = 'YOLOXWarmCosLR',  
                                 warmup_epochs=5,
                                 num_iters_per_epoch=num_iters_per_epoch,
                                 tot_num_epochs=epochs,
                                 min_lr_ratio=0.05,
                                 warmup_lr_start=0,
                                 steps_at_iteration=[50000],
                                 reduction_at_step=0.5,),
             }['YWC']

# print(scheduler['type'])
warmup= [None, dict(type = 'SquareWarmup', warmup_period=1000, scale=0.1),][0]

seg_loss_weight = 1.0
eval_ep = 5
save_ep = 50

bbox_order = 'cxywh'
bbox_norm = False
img_mean=[75.3, 76.6, 77.6]
img_std=[50.5, 53.8, 54.3]

train_process = [
    dict(type='GenerateEventBox',
        transforms = (          
            dict(
                name = 'Affine',
                parameters = dict(
                    translate_px = dict(
                        x = (-25, 25),
                        y = (-10, 10)
                    ),
                    rotate=(-6, 6),
                    scale=(0.85, 1.15),
                    cval=168,
                )
            ),
            dict(
                name = 'HorizontalFlip',
                parameters = dict(
                    p=0.5
                ),
            ),
            dict(
                name = 'ContrastNormalization', # 对比度增强
                parameters = dict(
                    alpha=(0.5, 1.5)
                ),
            ),
            dict(
                name = 'AddToHueAndSaturation', # 色调和饱和度的调整
                parameters = dict(
                    value=(-15, 15),
                    per_channel=True
                ),
            ),
        ),
        wh = (img_w, img_h),
        nb_classes=num_classes,
        bbox_order=bbox_order, 
        norm_box=bbox_norm,
        mean=img_mean,
        std=img_std,
    ),
    dict(type='ToTensor', keys=['img','bounding_box']),
]

val_process = [
    dict(type='GenerateEventBox',
        wh = (img_w, img_h),
        nb_classes=num_classes,
        bbox_order=bbox_order, 
        mean=img_mean,
        std=img_std,
    ),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/dsec'
dataset_type = 'DsecDetection'
mosaic = dict(mosaic_prob=0.6,
              enable_mixup=True,
              mixup_prob=0.9,
              mixup_scale=(0.5, 1.5))


dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
        mosaic=mosaic,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='val',
        processes=val_process,
        coco_eval=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
        coco_eval=True,
    )
)

workers = 16 
log_interval = num_iters_per_epoch//10 #50
lr_update_by_epoch=False
