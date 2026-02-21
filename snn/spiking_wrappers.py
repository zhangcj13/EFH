import warnings
from typing import List, Optional
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

# from detection.utils.torch_utils import fuse_conv_and_bn,fuse_scale2conv
# from detection.utils.quantity import SymmetryQ_Para

from .norm import get_norm
from .neuro import get_neuro
import copy
# from detection.utils.syops import spike_rate

class SpikConv(nn.Module):
    """Standard Convolutional Block"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        norm: dict = {'type':'BN'},
        neuro: dict = {'type':'IF'},
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = get_norm(norm, num_features=out_channels)
        self.act = get_neuro(neuro)
        # self.act.__class__.__name__
      
        
    def forwardn(self, x: torch.Tensor) -> torch.Tensor:
        # x0 = copy.deepcopy(x)
        
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)

        # if hasattr(self,'i_conv'):
        #     x0 = self.i_conv(x0)
        #     x0 = self.i_act(x0)

        #     diff = torch.abs(x-x0)
        #     print(torch.sum(diff))
        
        return x
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        y = self.conv(x)
        if self.norm:
            y = self.norm(y)
        y = self.act(y)

        return y


class NonSpikConv(SpikConv):
    """Standard Convolutional Block"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        norm: dict = {'type':'None'},
        neuro: dict = {'type':'NonSpikeIF'},
    ) -> None:
        super(NonSpikConv,self).__init__(in_channels,out_channels,kernel_size,stride,padding,groups,dilation,bias,norm,neuro)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=SpikConv.forward(self,x)
        if hasattr(self,'scale'):
            x=x*self.scale
        return x
