stage1=[True,False][0]  #  True False
exp_file='./configs/yolox_exp/RsHN_DSEC.py'
pre_weight=None
net_type = ['RsHNet','ResYOLOX'][0]
pre_sew=[True,False][1]
ckpt=None
input_size = (224, 320)
test_size = (224, 320)
cropped_height = 448

multiscale_range = 2
num_classes = 2

ts = 20 # 20 10 5
arch_net = {0:'resnet10', 
            1:'resnet18',
            2:'resnet34',
            3:'resnet50',
            11:'resnet18t',
            21:'resnet34t',
            }[11]

total_time_steps = ts
evt_dt = {10: 5000,
           5:10000,
          20: 2500,}[ts]

eval_us = ts*evt_dt

batch_size = { 5: 32,
              10: 16, # 32, 16,
              20: 8}[ts]

batch_size = batch_size if arch_net !='resnet50' else int(batch_size*0.5) #0.38
batch_size = int(batch_size*2) if arch_net =='resnet10' else batch_size #0.38

if net_type =='ResYOLOX':
    batch_size = batch_size*4

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

event_backbone ={0: dict(type='SewResnetSDRNN', 
                         arch = arch_net, 
                         in_channels=2,    
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],),
                1: dict(type='VanillaResNetEmd',
                        arch = arch_net,
                        in_channels=2,
                        norm = {'type':'BN'},
                        neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},
                        out_features=['layer2', 'layer3', 'layer4'],),
                2: dict(type='SewResnetSDRNN2', 
                         arch = arch_net, #'resnet18',
                         in_channels=2, 
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],), 
                21: dict(type='msSewResnetSDRNN', 
                         arch = arch_net, #'resnet18',
                         in_channels=2, 
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],), 
                3: dict(type='SewResnetNRNN', 
                         arch = arch_net, 
                         in_channels=2,    
                         cnf = 'ADD',    
                         norm = {'type':'BN'},    
                         neuro={'type':'LIFNode','v_threshold':v_threshold,'surrogate_function':surrogate_function},  
                         out_features=['layer2', 'layer3', 'layer4'],), 
                4: dict(type='ResNetRNNE', 
                         arch = arch_net, 
                         in_channels=2, 
                         out_features=['layer2', 'layer3', 'layer4'],),   
                }[2] # 2
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
     100 : dict(type='KDFNFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=False,
               dfn = 'f2e', 
               act="silu",),
     101 : dict(type='CDFNFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=False,
               act="silu",),
      1 : dict(type='RDDFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=False,
               act="silu",),
      2 : dict(type='LSTMFPN',
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),
      3 : dict(type='SoftFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),
      4 : dict(type='CAFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),
      5 : dict(type='LiteDDFPN', 
               in_channels = [256,512,1024],
               hidden_dim = hidden_dim,
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),
      6 : dict(type='ReduceDDFPN', 
               in_channels = [256,512,1024],
               width=width,
               fchn=ReduceChn,
               schn=ReduceChn,
               fz_channels=ReduceChn,
               depth=0.33,
               depthwise=True,
               act="silu",),
      7 : dict(type='ReduceDDPM', 
               in_channels = [256,512,1024],
               width=width,
               fchn=ReduceChn,
               schn=ReduceChn,
               fz_channels=ReduceChn,
               depth=0.33,
               depthwise=True,
               act="silu",),
      }[101]

if net_type == 'ResYOLOX':
    neck ={0:dict(type='ReducePAFPN', 
               in_channels = [256,512,1024],
               width=width,
               fchn=ReduceChn,
               depthwise=True,
               act="silu",),
           1:dict(type='ReduceFI', 
               in_channels = [256,512,1024],
               width=width,
               fchn=ReduceChn,
            #    depth=0.33,
               depthwise=True,
               act="silu",),
           2:dict(type='PAFPN', 
               in_channels = [256,512,1024],
               width=width,
               depth=0.33,
               depthwise=True,
               act="silu",),}[2]

fuze_channels = [256, 512, 1024]
if neck['type'] =='LiteDDFPN':
    fuze_channels =[hidden_dim*2, hidden_dim*2, hidden_dim*2]
elif neck['type'] =='ReduceDDFPN'  or neck['type'] =='ReduceDDPM':
    fuze_channels =[neck['fz_channels'][0]*2, neck['fz_channels'][1]*2, neck['fz_channels'][2]*2]
elif neck['type'] =='ReducePAFPN' or neck['type'] =='ReduceFI':
    fuze_channels =[neck['fchn'][0]*2, neck['fchn'][1]*2, neck['fchn'][2]*2]

# data for box detect
max_boxes = 256

bbox_heads = dict(type='YOLOX_Head',
                    num_classes=num_classes,
                    width=width, # 0.5
                    strides=[8, 16, 32],
                    in_channels=fuze_channels,
                    act="silu",
                    depthwise=False)


# train params
if stage1:
    warmup_epochs = 10 #10 20
    max_epoch = 160 # 70 90
    no_aug_epochs = 15 # 15

    no_event_epoch= 0 # 15  max_epoch+1
    freeze_image_backbone_epochs = [0, 20] #[15, 30]

    print_interval = 50 #50
    eval_interval = 5 # 10
    lr_all= 0.01 #0.01
else:
    warmup_epochs = 5 
    max_epoch = 50 
    no_aug_epochs = 15 

    print_interval = 50 
    eval_interval = 5 
    lr_all= 0.01 * 0.4

    # warmup_epochs = 1 
    # max_epoch = 20 
    # no_aug_epochs = 1 

    # print_interval = 10
    # eval_interval = 2
    # lr_all= 0.01 


# train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
#                         min_bbox_diag=15, min_bbox_height=10)
# test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
#                     min_bbox_diag=15, min_bbox_height=10)

#  dataset 
data_root = '/root/data1/ws/SNN_CV/data/dsec'
train_scale=2
val_scale=2
min_bbox_diag=15
min_bbox_height=10
train_min_bbox_scale=4
max_labels= 50
enable_mosaic=True

# ln -s /root/data1/dataset/OpenLane-V2 ./data/