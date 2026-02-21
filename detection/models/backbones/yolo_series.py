import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Callable

from detection.models.registry import BACKBONES
import numpy as np
import os
# from detection.utils.torch_utils import fuse_conv_and_bn
# from snn.qcfs.modules import ScaledNeuron

from  yolox.models.yolo_pafpn import YOLOPAFPN
from  yolox.models.yolo_fpn import YOLOFPN

@BACKBONES.register_module
class YOLOWrapper(nn.Module):
    def __init__(self,
                yolo = 'YOLOPAFPN',
                depth=1.0,
                width=1.0,
                in_features=("dark3", "dark4", "dark5"),
                in_channels=[256, 512, 1024],
                depthwise=False,
                act="silu",
                cfg=None):
        super(YOLOWrapper, self).__init__()
        self.cfg = cfg

        self.cfg = cfg
        if yolo=='YOLOPAFPN':
            self.model = YOLOPAFPN(depth, width, in_features, in_channels, 
                                   depthwise,act)
        elif yolo=='YOLOFPN':
            self.model = YOLOFPN(depth, in_features)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        x = self.model(x)
        return x



if __name__ == '__main__':
    model_dir='/root/data1/ws/SNN_CV/pretrained/yolov8n-cls.pt'
    pretrained_net = torch.load(model_dir)

    print(pretrained_net)
