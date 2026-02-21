import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from detection.utils.torch_utils import fuse_conv_and_bn,fuse_conv_and_bn_list
from ..registry import NECKS
from snn.spiking_wrappers import SpikConv
from yolox.models.network_blocks import  BaseConv, CSPLayer, DWConv



@NECKS.register_module
class PAFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

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
    
    def forward(self, features):
        assert len(features)>=3

        [x2, x1, x0] = features[-3:]

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
class ReducePAFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 fchn: list=[64,128,256],
                 width=0.5,
                 depth=0.33,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ichn = (np.array(in_channels)*width).astype(np.int32)

        self.i_reduce0 = Conv(ichn[2], fchn[2], 3, 1, act=act )
        self.i_reduce1 = Conv(ichn[1], fchn[1], 3, 1, act=act )
        self.i_reduce2 = Conv(ichn[0], fchn[0], 3, 1, act=act )

        in_channels = [int(_c/width) for _c in fchn]

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
    
    def forward(self, features):
        assert len(features)>=3

        [x2, x1, x0] = features[-3:]

        x0 = self.i_reduce0(x0)
        x1 = self.i_reduce1(x1)
        x2 = self.i_reduce2(x2)

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
class ReduceFI(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 fchn: list=[64,128,256],
                 width=0.5,
                 depthwise=False,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ichn = (np.array(in_channels)*width).astype(np.int32)

        self.i_reduce0 = Conv(ichn[2], fchn[2], 3, 1, act=act )
        self.i_reduce1 = Conv(ichn[1], fchn[1], 3, 1, act=act )
        self.i_reduce2 = Conv(ichn[0], fchn[0], 3, 1, act=act )

    
    def forward(self, features):
        assert len(features)>=3

        [x2, x1, x0] = features[-3:]

        x0 = self.i_reduce0(x0)
        x1 = self.i_reduce1(x1)
        x2 = self.i_reduce2(x2)

        outputs = (x2, x1, x0)
        return outputs

@NECKS.register_module
class PAFPN64(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 depth=0.33,
                 width=1.0,
                 depthwise=False,
                 out_channel=64,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[3]), int(in_channels[2]*0.5), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(in_channels[2]*1.5),
            int(in_channels[2]*0.75),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[2]*0.75), int(in_channels[1] * 0.75), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(in_channels[1] * 1.75),
            int(in_channels[1]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv2 = BaseConv(
            int(in_channels[1]), int(in_channels[0]), 1, 1, act=act
        )
        self.C3_p2 = CSPLayer(
            int(in_channels[0] * 2.0),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # bottom-up conv
        self.bu_conv3 = Conv(
            out_channel, out_channel, 3, 2, act=act
        )
        self.C3_n2 = CSPLayer(
            int(out_channel+in_channels[0]),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            out_channel, out_channel, 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(out_channel+in_channels[1] * 0.75),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            out_channel, out_channel, 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(out_channel+in_channels[2]*0.5),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
    
    def forward(self, features):
        assert len(features)>=4

        [x3, x2, x1, x0] = features[-4:]

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out1 = self.C3_p3(f_out1)  # 512->256/8

        fpn_out2 = self.reduce_conv2(f_out1)  # 512->256/16
        f_out2 = self.upsample(fpn_out2)  # 256/8
        f_out2 = torch.cat([f_out2, x3], 1)  # 256->512/8
        pan_out3 = self.C3_p2(f_out2)  # 512->256/8

        p_out2 = self.bu_conv3(pan_out3)  # 256->256/16
        p_out2 = torch.cat([p_out2, fpn_out2], 1)  # 256->512/16
        pan_out2 = self.C3_n2(p_out2)  # 512->512/16

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        return outputs


@NECKS.register_module
class PA3FPN64(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 depth=0.33,
                 width=1.0,
                 depthwise=False,
                 out_channel=64,
                 act="relu",
                 cfg = None) -> None:
        super().__init__()
        self.cfg=cfg

        # self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2]), int(in_channels[1]*0.5), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(in_channels[1]*1.5),
            int(in_channels[0]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[0]), int(in_channels[0] * 0.5), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(in_channels[0] * 1.5),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            out_channel, out_channel, 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(out_channel+in_channels[0] * 0.5),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            out_channel, out_channel, 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(out_channel+in_channels[1]*0.5),
            out_channel,
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
    
    def forward(self, features):
        assert len(features)>=3

        [x2, x1, x0] = features[-3:]

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