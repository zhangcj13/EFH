import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from detection.utils.torch_utils import fuse_conv_and_bn,fuse_conv_and_bn_list
from ..registry import NECKS

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def fuse(self,):
         fuse_conv_and_bn_list(self.double_conv)

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@NECKS.register_module
class UpF(nn.Module):
    def __init__(self,
                 in_channels: list=[1024,256,128,64],
                 out_channels:list=[512,256,128,64],
                 bilinear:bool=True,
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.up_modules=[]
        for i,ichn in enumerate(in_channels):
            self.add_module(f'up{i}', Up(ichn, out_channels[i], bilinear))
            self.up_modules.append(f'up{i}')
    
    def forward(self, features):

        output=[]
        x=features[-1]
        for i, name in enumerate(self.up_modules):
            f = features[-2-i]
            layer = getattr(self, name)
            x = layer(x,f) 
            output.append(x)
       
        return output #output[::-1]

