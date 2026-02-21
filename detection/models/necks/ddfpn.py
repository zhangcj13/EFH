import sys
sys.path.append('/root/data1/ws/SNN_CV')

import warnings

import numpy as np
from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from detection.utils.torch_utils import fuse_conv_and_bn,fuse_conv_and_bn_list
from detection.models.registry import NECKS
from snn.spiking_wrappers import SpikConv
from yolox.models.network_blocks import  BaseConv, CSPLayer, DWConv,get_activation

from detection.models.backbones.sew_resnet_rnn import DWSConvLSTM2d

from detection.models.layers.dfn import DDPM, DDPMLSTM2d
from detection.models.layers.sam import SAM, CrossAttention
# from detection.models.layers.dynamic_conv import Dynamic_conv2d

def get_ksd(kernel_size):
    if kernel_size == 17:
        kernel_sizes = [5, 9, 3, 3, 3]
        dilates = [1, 2, 4, 5, 7]
    elif kernel_size == 15:
        kernel_sizes = [5, 7, 3, 3, 3]
        dilates = [1, 2, 3, 5, 7]
    elif kernel_size == 13:
        kernel_sizes = [5, 7, 3, 3, 3]
        dilates = [1, 2, 3, 4, 5]
    elif kernel_size == 11:
        kernel_sizes = [5, 5, 3, 3, 3]
        dilates = [1, 2, 3, 4, 5]
    elif kernel_size == 9:
        kernel_sizes = [5, 5, 3, 3]
        dilates = [1, 2, 3, 4]
    elif kernel_size == 7:
        kernel_sizes = [5, 3, 3]
        dilates = [1, 2, 3]
    elif kernel_size == 5:
        kernel_sizes = [3, 3]
        dilates = [1, 2]
    else:
        kernel_sizes = [kernel_size]
        dilates = [1]
        # raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

    return kernel_sizes,dilates

@NECKS.register_module
class SoftFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        schn=[int(chn * width) for chn in in_channels]

        self.fuseConv0 = Conv(schn[2]*2, schn[2], 3, 1, act=act )
        self.fuseConv1 = Conv(schn[1]*2, schn[1], 3, 1, act=act )
        self.fuseConv2 = Conv(schn[0]*2, schn[0], 3, 1, act=act )

    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # msf=sf
        [f2, f1, f0] = ff[-3:]
        [s2, s1, s0] =ssf[-3:]
        
        x0 = self.fuseConv0(torch.cat([f0, s0], 1))
        x1 = self.fuseConv1(torch.cat([f1, s1], 1))
        x2 = self.fuseConv2(torch.cat([f2, s2], 1))

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


@NECKS.register_module
class DDFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 dfn = 'e2f',  # [e2f,f2e]
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        schn=[int(chn * width) for chn in in_channels]
        self.selfdc_32 = DDPM(schn[2], schn[2], schn[2], 3, 4)
        self.selfdc_16 = DDPM(schn[1], schn[1], schn[1], 3, 4)
        self.selfdc_8  = DDPM(schn[0], schn[0], schn[0], 3, 4)

        self.dfn=dfn
        if self.dfn=='f2e':
            print('------------- current feature dynamic filter is the frame feature -------------')

    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # msf=sf
        [f2, f1, f0] = ff[-3:]
        [s2, s1, s0] =ssf[-3:]

        if self.dfn=='f2e':
            x0 = self.selfdc_32(s0,f0)
            x1 = self.selfdc_16(s1,f1)
            x2 = self.selfdc_8( s2,f2)
        else:
            x0 = self.selfdc_32(f0,s0)
            x1 = self.selfdc_16(f1,s1)
            x2 = self.selfdc_8( f2,s2)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


@NECKS.register_module
class LiteDDFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 hidden_dim: int = 64,
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.f_reduce0 = Conv(int(in_channels[2] * width), hidden_dim, 3, 1, act=act )
        self.f_reduce1 = Conv(int(in_channels[1] * width), hidden_dim, 3, 1, act=act )
        self.f_reduce2 = Conv(int(in_channels[0] * width), hidden_dim, 3, 1, act=act )

        self.s_reduce0 = Conv(int(in_channels[2] * width), hidden_dim, 3, 1, act=act )
        self.s_reduce1 = Conv(int(in_channels[1] * width), hidden_dim, 3, 1, act=act )
        self.s_reduce2 = Conv(int(in_channels[0] * width), hidden_dim, 3, 1, act=act )

        self.selfdc_32 = DDPM(hidden_dim, hidden_dim, hidden_dim, 3, 4)
        self.selfdc_16 = DDPM(hidden_dim, hidden_dim, hidden_dim, 3, 4)
        self.selfdc_8  = DDPM(hidden_dim, hidden_dim, hidden_dim, 3, 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            hidden_dim, hidden_dim, 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            2 * hidden_dim,
            hidden_dim,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            hidden_dim, hidden_dim, 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            2 * hidden_dim,
            hidden_dim,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            hidden_dim, hidden_dim, 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            2*hidden_dim,
            hidden_dim,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            hidden_dim, hidden_dim, 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            2*hidden_dim,
            hidden_dim,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )



    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # reduce chn of image feature
        [f2, f1, f0] = ff[-3:]
        f0 = self.f_reduce0(f0)
        f1 = self.f_reduce1(f1)
        f2 = self.f_reduce2(f2)
        # reduce chn of event feature
        [s2, s1, s0] =ssf[-3:]
        s0 = self.s_reduce0(s0)
        s1 = self.s_reduce1(s1)
        s2 = self.s_reduce2(s2)      
        # DFN fuse features
        x0 = self.selfdc_32(f0,s0)
        x1 = self.selfdc_16(f1,s1)
        x2 = self.selfdc_8( f2,s2)

        fpn_out0 = self.lateral_conv0(x0)  # 64->64/32
        f_out0 = self.upsample(fpn_out0)  # 64/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 64->128/16
        f_out0 = self.C3_p4(f_out0)  # 128->64/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 64->64/16
        f_out1 = self.upsample(fpn_out1)  # 64/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 64->128/8
        pan_out2 = self.C3_p3(f_out1)  # 128->64/8

        p_out1 = self.bu_conv2(pan_out2)  # 64->64/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 64->128/16
        pan_out1 = self.C3_n3(p_out1)  # 128->64/16

        p_out0 = self.bu_conv1(pan_out1)  # 64->64/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 64->128/32
        pan_out0 = self.C3_n4(p_out0)  # 128->64/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


@NECKS.register_module
class RDDFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.selfdc_32 = DDPMLSTM2d(512, 512, 512, 3, 4)
        self.selfdc_16 = DDPMLSTM2d(256, 256, 256, 3, 4)
        self.selfdc_8  = DDPMLSTM2d(128, 128, 128, 3, 4)

    def forward(self, features, spike_fea, seq_length):
        [f2, f1, f0] = features[-3:]
        [ss2, ss1, ss0] = spike_fea[-3:]

        B,_,_,_,T = ss0.shape

        for  n in range(0,T):
            if n ==0:
                hc0=(f0, torch.zeros_like(ss0[...,0]))
                hc1=(f1, torch.zeros_like(ss1[...,0]))
                hc2=(f2, torch.zeros_like(ss2[...,0]))
                h0s=torch.zeros_like(hc0[0]).unsqueeze_(-1).repeat(1,1,1,1, T)
                h1s=torch.zeros_like(hc1[0]).unsqueeze_(-1).repeat(1,1,1,1, T)
                h2s=torch.zeros_like(hc2[0]).unsqueeze_(-1).repeat(1,1,1,1, T)

            hc0 = self.selfdc_32(ss0[...,n],hc0)
            hc1 = self.selfdc_16(ss1[...,n],hc1)
            hc2 = self.selfdc_8( ss2[...,n],hc2)

            h0s[...,n]=hc0[0]
            h1s[...,n]=hc1[0]
            h2s[...,n]=hc2[0]

        xsf=[]
        for s in [h0s,h1s,h2s]:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            xsf.append(rnn_output)

        x0,x1,x2 = xsf

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


@NECKS.register_module
class LSTMFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.lstm0 = DWSConvLSTM2d(dim = 512, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.lstm1 = DWSConvLSTM2d(dim = 256, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.lstm2 = DWSConvLSTM2d(dim = 128, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)

        self.fuseConv0 = Conv(512*2, 512, 3, 1, act=act )
        self.fuseConv1 = Conv(256*2, 256, 3, 1, act=act )
        self.fuseConv2 = Conv(128*2, 128, 3, 1, act=act )


    def forward(self, features, spike_fea, seq_length):
        [f2, f1, f0] = features[-3:]
        [ss2, ss1, ss0] = spike_fea[-3:]

        B,_,_,_,TS = ss0.shape

        for t in range(TS):
            if t==0:
                hc0=(torch.zeros_like(f0), torch.zeros_like(f0))
                hc1=(torch.zeros_like(f1), torch.zeros_like(f1))
                hc2=(torch.zeros_like(f2), torch.zeros_like(f2))
                h0s=torch.zeros_like(hc0[0]).unsqueeze_(-1).repeat(1,1,1,1, TS)
                h1s=torch.zeros_like(hc1[0]).unsqueeze_(-1).repeat(1,1,1,1, TS)
                h2s=torch.zeros_like(hc2[0]).unsqueeze_(-1).repeat(1,1,1,1, TS)

            cf0 = torch.cat([f0, ss0[...,t]], 1)
            cf1 = torch.cat([f1, ss1[...,t]], 1)
            cf2 = torch.cat([f2, ss2[...,t]], 1)
            cf0 = self.fuseConv0(cf0)
            cf1 = self.fuseConv1(cf1)
            cf2 = self.fuseConv2(cf2)
            hc0 = self.lstm0(cf0,hc0)
            hc1 = self.lstm1(cf1,hc1)
            hc2 = self.lstm2(cf2,hc2)

            h0s[...,t]=hc0[0]
            h1s[...,t]=hc1[0]
            h2s[...,t]=hc2[0]
            
        xsf=[]
        for s in [h0s,h1s,h2s]:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            xsf.append(rnn_output)

        x0,x1,x2 = xsf

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs

@NECKS.register_module
class CAFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.CrossAtt0 = CrossAttention(512,single_out=True)
        self.CrossAtt1 = CrossAttention(256,single_out=True)
        self.CrossAtt2 = CrossAttention(128,single_out=True)

    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # msf=sf
        [f2, f1, f0] = ff[-3:]
        [s2, s1, s0] = ssf[-3:]
        
        x0 = self.CrossAtt0(s0,f0)
        x1 = self.CrossAtt1(s1,f1)
        x2 = self.CrossAtt2(s2,f2)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs

@NECKS.register_module
class ReduceDDFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 fchn: list=[64,128,256],
                 schn: list=[64,128,256],
                 fz_channels: list=[64,128,256],
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ichn = (np.array(in_channels)*width).astype(np.int32)
  
        self.f_reduce0 = Conv(ichn[2], fchn[2], 3, 1, act=act )
        self.f_reduce1 = Conv(ichn[1], fchn[1], 3, 1, act=act )
        self.f_reduce2 = Conv(ichn[0], fchn[0], 3, 1, act=act )

        self.s_reduce0 = Conv(ichn[2], schn[2], 3, 1, act=act )
        self.s_reduce1 = Conv(ichn[1], schn[1], 3, 1, act=act )
        self.s_reduce2 = Conv(ichn[0], schn[0], 3, 1, act=act )

        self.selfdc_32 = DDPM(fchn[2], schn[2], fz_channels[2], 3, 4)
        self.selfdc_16 = DDPM(fchn[1], schn[1], fz_channels[1], 3, 4)
        self.selfdc_8  = DDPM(fchn[0], schn[0], fz_channels[0], 3, 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            fz_channels[2], fz_channels[1], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            2 * fz_channels[1],
            fz_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            fz_channels[1], fz_channels[0], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            2 * fz_channels[0],
            fz_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            fz_channels[0], fz_channels[0], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            2*fz_channels[0],
            fz_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            fz_channels[1], fz_channels[1], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            2*fz_channels[1],
            fz_channels[2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )



    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # reduce chn of image feature
        [f2, f1, f0] = ff[-3:]
        f0 = self.f_reduce0(f0)
        f1 = self.f_reduce1(f1)
        f2 = self.f_reduce2(f2)
        # reduce chn of event feature
        [s2, s1, s0] =ssf[-3:]
        s0 = self.s_reduce0(s0)
        s1 = self.s_reduce1(s1)
        s2 = self.s_reduce2(s2)      
        # DFN fuse features
        x0 = self.selfdc_32(f0,s0)
        x1 = self.selfdc_16(f1,s1)
        x2 = self.selfdc_8( f2,s2)

        fpn_out0 = self.lateral_conv0(x0)  # 64->64/32
        f_out0 = self.upsample(fpn_out0)  # 64/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 64->128/16
        f_out0 = self.C3_p4(f_out0)  # 128->64/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 64->64/16
        f_out1 = self.upsample(fpn_out1)  # 64/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 64->128/8
        pan_out2 = self.C3_p3(f_out1)  # 128->64/8

        p_out1 = self.bu_conv2(pan_out2)  # 64->64/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 64->128/16
        pan_out1 = self.C3_n3(p_out1)  # 128->64/16

        p_out0 = self.bu_conv1(pan_out1)  # 64->64/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 64->128/32
        pan_out0 = self.C3_n4(p_out0)  # 128->64/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


@NECKS.register_module
class ReduceDDPM(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 fchn: list=[64,128,256],
                 schn: list=[64,128,256],
                 fz_channels: list=[64,128,256],
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ichn = (np.array(in_channels)*width).astype(np.int32)
  
        self.f_reduce0 = Conv(ichn[2], fchn[2], 3, 1, act=act )
        self.f_reduce1 = Conv(ichn[1], fchn[1], 3, 1, act=act )
        self.f_reduce2 = Conv(ichn[0], fchn[0], 3, 1, act=act )

        self.s_reduce0 = Conv(ichn[2], schn[2], 3, 1, act=act )
        self.s_reduce1 = Conv(ichn[1], schn[1], 3, 1, act=act )
        self.s_reduce2 = Conv(ichn[0], schn[0], 3, 1, act=act )

        self.selfdc_32 = DDPM(fchn[2], schn[2], fz_channels[2], 3, 4)
        self.selfdc_16 = DDPM(fchn[1], schn[1], fz_channels[1], 3, 4)
        self.selfdc_8  = DDPM(fchn[0], schn[0], fz_channels[0], 3, 4)

        
    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # reduce chn of image feature
        [f2, f1, f0] = ff[-3:]
        f0 = self.f_reduce0(f0)
        f1 = self.f_reduce1(f1)
        f2 = self.f_reduce2(f2)
        # reduce chn of event feature
        [s2, s1, s0] =ssf[-3:]
        s0 = self.s_reduce0(s0)
        s1 = self.s_reduce1(s1)
        s2 = self.s_reduce2(s2)      
        # DFN fuse features
        x0 = self.selfdc_32(f0,s0)
        x1 = self.selfdc_16(f1,s1)
        x2 = self.selfdc_8( f2,s2)

        outputs = (x2, x1, x0)
        return outputs

class VxVDFN(nn.Module):
    def __init__(self,
                 in_xC, in_yC, out_C, 
                 kernel_size=5,
                 act="relu",
                 ):
        super().__init__()

        self.kernel_sizes, self.dilates = get_ksd(kernel_size)
       
        for k, r in zip(self.kernel_sizes, self.dilates):           
            self.__setattr__('gernerate_kernel_k{}_{}'.format(k, r), nn.Sequential( 
                nn.Conv2d(in_channels=in_yC, out_channels=in_yC, kernel_size=k, stride=1,padding=(r * (k - 1) + 1) // 2, dilation=r, groups=in_yC, bias=False),
                nn.BatchNorm2d(in_yC),
                get_activation(act, inplace=True),
                nn.Conv2d(in_channels=in_yC, out_channels=in_xC*k**2, kernel_size=1),
                # nn.BatchNorm2d(in_xC*k**2),
                # get_activation(act, inplace=True),
                ))             

            self.__setattr__('unfold_k{}_{}'.format(k, r), nn.Unfold(kernel_size=k, dilation=r, padding=(r * (k - 1) + 1) // 2, stride=1))
        
        self.norm = nn.Sequential(nn.BatchNorm2d(in_yC),get_activation(act, inplace=True))


    def forward(self, y : torch.Tensor, x: Optional[torch.Tensor] = None):
        ''' use input event y to refine input feature x '''
        
        N, xC, xH, xW = x.size()
    
        kernels = {}

        combined=0
        for k, r in zip(self.kernel_sizes, self.dilates):
            gk_k = self.__getattr__('gernerate_kernel_k{}_{}'.format(k, r))
            unfold_k = self.__getattr__('unfold_k{}_{}'.format(k, r))

            kernel = gk_k(y).reshape([N, xC, k ** 2, xH, xW])

            kernels['{}_{}'.format(k, r)] = kernel

            unfold_x = unfold_k(x).reshape([N, xC, -1, xH, xW])
            result = (unfold_x * kernel).sum(2)

            combined += result
        
        combined = self.norm(combined)
                
        return combined
    

@NECKS.register_module
class KDFNFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 dfn = 'e2f',  # [e2f,f2e]
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        schn=[int(chn * width) for chn in in_channels]
 
        # self.selfdc_32 = VxVDFN(schn[2], schn[2], schn[2], 5, act=act)
        # self.selfdc_16 = VxVDFN(schn[1], schn[1], schn[1], 7, act=act)
        # self.selfdc_8  = VxVDFN(schn[0], schn[0], schn[0], 11, act=act)

        self.selfdc_32 = VxVDFN(schn[2], schn[2], schn[2], 3, act=act)
        self.selfdc_16 = VxVDFN(schn[1], schn[1], schn[1], 5, act=act)
        self.selfdc_8  = VxVDFN(schn[0], schn[0], schn[0], 7, act=act)

        self.dfn=dfn
        if self.dfn=='f2e':
            print('------------- current feature dynamic filter is the frame feature -------------')

    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # msf=sf
        [f2, f1, f0] = ff[-3:]
        [s2, s1, s0] =ssf[-3:]

        if self.dfn=='f2e':
            x0 = self.selfdc_32(s0,f0)
            x1 = self.selfdc_16(s1,f1)
            x2 = self.selfdc_8( s2,f2)
        else:
            x0 = self.selfdc_32(f0,s0)
            x1 = self.selfdc_16(f1,s1)
            x2 = self.selfdc_8( f2,s2)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


class CrossDFN(nn.Module):
    def __init__(self,
                 in_xC, in_yC, out_C, 
                 kernel_size=5,
                 act="relu",
                 ):
        super().__init__()

        self.kernel_sizes, self.dilates = get_ksd(kernel_size)
       
        for k, r in zip(self.kernel_sizes, self.dilates):           
            self.__setattr__('gernerate_kernel_yk{}_{}'.format(k, r), nn.Sequential( 
                nn.Conv2d(in_channels=in_yC, out_channels=in_yC, kernel_size=k, stride=1,padding=(r * (k - 1) + 1) // 2, dilation=r, groups=in_yC, bias=False),
                nn.BatchNorm2d(in_yC),
                get_activation(act, inplace=True),
                nn.Conv2d(in_channels=in_yC, out_channels=in_xC*k**2, kernel_size=1),
                # nn.BatchNorm2d(in_xC*k**2),
                # get_activation(act, inplace=True),
                ))      

            self.__setattr__('gernerate_kernel_xk{}_{}'.format(k, r), nn.Sequential( 
                nn.Conv2d(in_channels=in_xC, out_channels=in_xC, kernel_size=k, stride=1,padding=(r * (k - 1) + 1) // 2, dilation=r, groups=in_xC, bias=False),
                nn.BatchNorm2d(in_xC),
                get_activation(act, inplace=True),
                nn.Conv2d(in_channels=in_xC, out_channels=in_yC*k**2, kernel_size=1),
                ))             

            self.__setattr__('unfold_k{}_{}'.format(k, r), nn.Unfold(kernel_size=k, dilation=r, padding=(r * (k - 1) + 1) // 2, stride=1))
        
        self.norm_x = nn.Sequential(nn.BatchNorm2d(in_yC),get_activation(act, inplace=True))
        self.norm_y = nn.Sequential(nn.BatchNorm2d(in_xC),get_activation(act, inplace=True))

        self.fuse = BaseConv(in_yC + in_xC, out_C, 3, 1, 1, act=act)

    def forward(self, y : torch.Tensor, x: Optional[torch.Tensor] = None):
        ''' use input event y to refine input feature x '''
        
        N, xC, xH, xW = x.size()
        _, yC, yH, yW = y.size()

        assert xC==yC and xH==yH and xW==yW, 'the spatial size of c, x and y should be the same!'
    
        kernels_y = {}
        kernels_x = {}

        combined_x=0
        combined_y=0
        for k, r in zip(self.kernel_sizes, self.dilates):
            gk_yk = self.__getattr__('gernerate_kernel_yk{}_{}'.format(k, r))
            gk_xk = self.__getattr__('gernerate_kernel_xk{}_{}'.format(k, r))
            unfold_k = self.__getattr__('unfold_k{}_{}'.format(k, r))

            kernel_y = gk_yk(y).reshape([N, xC, k ** 2, xH, xW])
            kernels_y['{}_{}'.format(k, r)] = kernel_y

            kernel_x = gk_xk(x).reshape([N, xC, k ** 2, xH, xW])
            kernels_x['{}_{}'.format(k, r)] = kernel_x

            unfold_x = unfold_k(x).reshape([N, xC, -1, xH, xW])
            result_x = (unfold_x * kernel_y).sum(2)

            unfold_y = unfold_k(y).reshape([N, xC, -1, xH, xW])
            result_y = (unfold_y * kernel_x).sum(2)

            combined_x += result_x
            combined_y += result_y
            
        
        combined_x = self.norm_x(combined_x)
        combined_y = self.norm_y(combined_y)

        fused = self.fuse(torch.cat((combined_x, combined_y), dim=1))
                
        return fused


@NECKS.register_module
class CDFNFPN(nn.Module):
    def __init__(self,
                 in_channels: list=[256,512,1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        self.fea_num = len(in_channels)
        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        schn = [int(chn * width) for chn in in_channels]
 
        # kk_size=[3,3,3]
        # kk_size=[7,5,3]
        # kk_size=[5,5,5]
        kk_size = [9,7,5]
        # kk_size=[9,9,9]
        self.selfdc_32 = CrossDFN(schn[2], schn[2], schn[2], kk_size[2], act=act)
        self.selfdc_16 = CrossDFN(schn[1], schn[1], schn[1], kk_size[1], act=act)
        self.selfdc_8  = CrossDFN(schn[0], schn[0], schn[0], kk_size[0], act=act)

    def forward(self, features, spike_fea, seq_length):
        ff=features[-self.fea_num:]
        sf=spike_fea[-self.fea_num:]
        ssf=[]
        for s in sf:
            B,C,H,W,T=s.shape
            last_step_index_list = (seq_length).view(-1, 1, 1, 1).expand(B, C, H, W).unsqueeze(4)
            rnn_output = s.gather(4, last_step_index_list).squeeze(4)
            ssf.append(rnn_output)
        # msf=sf
        [f2, f1, f0] = ff[-3:]
        [s2, s1, s0] =ssf[-3:]

        x0 = self.selfdc_32(f0,s0)
        x1 = self.selfdc_16(f1,s1)
        x2 = self.selfdc_8( f2,s2)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs


if __name__ == '__main__':
    # x = torch.randn(24, 8,  16,16)
    # print(x.shape)

    net = CDFNFPN(in_channels = [256,512,1024],
                width=0.25,
                depth=0.33,
                depthwise=False,
                act="silu",)
    print(net)

    features = [torch.randn(1, 64, 64,64),
                torch.randn(1, 128, 32,32),
                torch.randn(1, 256,16,16)]
    spike_fea = [torch.randn(1, 64, 64,64,5),
                torch.randn(1, 128, 32,32,5),
                torch.randn(1, 256,16,16,5)]
    seq_length = torch.tensor([3])

    out = net(features, spike_fea, seq_length)