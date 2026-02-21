stage1=True
exp_file='./configs/yolox_exp/RsHN_PKUSOD.py'
pre_weight=None
net_type = 'RsHNet'
pre_sew=[True,False][1]
ckpt=[None,
      ][0]
input_size = (256, 352)
test_size = (256, 352)

multiscale_range = 2
num_classes = 3

ts = 8 
arch_net = {0:'resnet10', 
            1:'resnet18',
            2:'resnet34',
            3:'resnet50',
            11:'resnet18t',
            21:'resnet34t',
            }[21]

total_time_steps = ts
evt_dt = {8: 4850,
          3: 11433,
         14: 2450,}[ts]

eval_us = ts*evt_dt

batch_size = { 3: 32,
               8: 32, # 32, 16,
              14: 8}[ts]


featuremap_out_channel=[64, 128, 256, 512] if arch_net not in ['resnet10','resnet18t','resnet34t'] else [64, 64, 128, 256]

frame_backbone = dict(
    type='ResNetWrapper',
    resnet=arch_net,
    pretrained=True,
    in_channels=featuremap_out_channel,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

v_threshold = 0.5 #1.0 0.5
if stage1:
    surrogate_function = ['PseudoSpike','Atan','Sigmoid'][0]
else:
    surrogate_function = ['PseudoSpike','Atan','Sigmoid'][1]

event_backbone ={1: dict(type='SewResnetSDRNN2', 
                         arch = arch_net, #'resnet18',
                         in_channels=2, 
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],), 
                2: dict(type='SewResnetNRNN', 
                         arch = arch_net, 
                         in_channels=2,    
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],),  
                }[1]
hidden_dim = {'resnet10':64,
              'resnet18':64,
              'resnet18t':64,
              'resnet34':128,
              'resnet34t':128,
              'resnet50':256}[arch_net]

width  = {'resnet10':0.25,
          'resnet18t':0.25,
          'resnet34t':0.25,
          'resnet18':0.5,
          'resnet34':0.5,
          'resnet50':2.0}[arch_net]

ReduceChn={'resnet10':[32,64,128],
           'resnet18t':[32,64,128],
           'resnet18':[32,64,128], # [32,64,128] [64,128,256]
           'resnet34':[32,64,128],
           'resnet34t':[32,64,128],
           'resnet50':[32,64,128]}[arch_net]

neck={0 : dict(type='DDFPN', # RDDFPN DDFPN
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=False,
               dfn = 'f2e',  # [e2f,f2e]
               act="silu",),
    101 : dict(type='CDFNFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=False,
               act="silu",),
      3 : dict(type='SoftFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),
      }[101]

if net_type == 'ResYOLOX':
    neck ={2:dict(type='PAFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),}[2]

fuze_channels = [256, 512, 1024]

# data for box detect
max_boxes = 256

bbox_heads = {0: dict(type='YOLOX_Head',
                        num_classes=num_classes,
                        width=width, # 0.5
                        strides=[8, 16, 32],
                        in_channels=fuze_channels,
                        act="silu",
                        depthwise=False),}[0]


# train params
if stage1:
    warmup_epochs = 5 #10 20
    max_epoch = 100 # 70 90
    no_aug_epochs = 15 # 15

    no_event_epoch= 0 # 15  max_epoch+1
    freeze_image_backbone_epochs = [0, 15] #[15, 30]

    print_interval = 50 #50
    eval_interval = 5 # 10
    lr_all= 0.01 #0.01
else:
    warmup_epochs = 5 
    max_epoch = 50 
    no_aug_epochs = 15 

    print_interval = 50 
    eval_interval = 5 
    lr_all= 0.01


# train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
#                         min_bbox_diag=15, min_bbox_height=10)
# test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
#                     min_bbox_diag=15, min_bbox_height=10)

#  dataset 
data_root = '/root/data1/ws/SNN_CV/data/PKU_DAVIS_SOD/'
train_scale=2
val_scale=2
min_bbox_diag=7 #15
min_bbox_height=5 #10
train_min_bbox_scale=4
max_labels= 50
enable_mosaic = True

# ln -s /root/data1/dataset/OpenLane-V2 ./data/