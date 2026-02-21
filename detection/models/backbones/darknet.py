import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Callable

from detection.models.registry import BACKBONES
import numpy as np
import os
from detection.utils.torch_utils import fuse_conv_and_bn
from snn.qcfs.modules import ScaledNeuron


USE_RELU = True  # replace LeakyReLU by Relu for snn conversion

class Conv(nn.Module):
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
        inplace: bool = True,
        bias: bool = False,
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
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if USE_RELU:
            self.act = nn.ReLU(inplace=inplace)
        else:
            # self.act = nn.LeakyReLU(0.01, inplace=inplace)
            self.act = nn.LeakyReLU(0.1)
    
    def fuse(self,):
        self.conv = fuse_conv_and_bn(self.conv,self.norm)
        self.norm = None

        # if 'Neuron' in self.act.__class__.__name__:
        #     module.reset()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, in_planes: int, shortcut: bool = True) -> None:
        super().__init__()
        planes = in_planes // 2
        self.shortcut = shortcut
        self.conv1 = Conv(in_channels=in_planes, out_channels=planes, padding=0)
        self.conv2 = Conv(in_channels=planes, out_channels=in_planes, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out += residual
        return out


class CSP(nn.Module):
    """Cross Stage Partial Block <https://arxiv.org/pdf/1911.11929.pdf>"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        e: float = 0.5
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * e)  # hidden channels

        self.conv1 = Conv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=1, stride=1,
                               bias=False)
        self.conv4 = Conv(in_channels=2 * mid_channels, out_channels=out_channels, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(num_features=2 * mid_channels)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[ResidualBlock(mid_channels, shortcut=shortcut) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Elastic(nn.Module):
    """Elastic Block <https://arxiv.org/abs/1812.05262>"""

    def __init__(self, in_planes: int) -> None:
        super().__init__()
        mid_planes = in_planes // 2

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = Conv(in_channels=in_planes, out_channels=mid_planes // 2, padding=0)
        self.conv2 = Conv(in_channels=mid_planes // 2, out_channels=in_planes, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        elastic = x

        # check the input size before downsample
        if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
            elastic = F.pad(elastic, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate')

        down = self.down(elastic)
        elastic = self.conv1(down)
        elastic = self.conv2(elastic)
        up = self.up(elastic)
        # check the output size after upsample
        if up.size(2) > x.size(2) or up.size(3) > x.size(3):
            up = up[:, :, :x.size(2), :x.size(3)]

        half = self.conv1(x)
        half = self.conv2(half)

        out = up + half  # elastic add
        out += residual  # residual add

        return out


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


def _initialize_weights(module):
    """Initialize the weights of convolutional, batch normalization, and linear layers"""

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)

class DarkNet7(nn.Module):

    def __init__(self, num_classes: int = 1000, init_weight: bool = True, extract_features:bool =True) -> None:
        super().__init__()

        if init_weight:
            self.apply(_initialize_weights)

        self.layer1 = nn.Sequential(
            Conv(in_channels=3, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=64, out_channels=128, kernel_size=3),

            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=128, out_channels=256, kernel_size=3),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=256, out_channels=512, kernel_size=3),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1),
            Conv(in_channels=512, out_channels=1024, kernel_size=3),
        )
        self.fmap_c = [256, 1024]
        self.stride = [16, 32]
        self.classifier = None
        if not extract_features:
            self.classifier = nn.Sequential(
                Conv(in_channels=1024, out_channels=num_classes, kernel_size=1),
                GlobalAvgPool2d()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        if self.classifier is not None:
            return self.classifier(f2)
        return [f1,f2]

class DarkNet7F(nn.Module):

    def __init__(self, num_classes: int = 1000, init_weight: bool = True, extract_features = ['layer1', 'layer2', 'layer3', 'layer4']) -> None:
        super().__init__()

        if init_weight:
            self.apply(_initialize_weights)
        
        self.layer1 = nn.Sequential(
            Conv(in_channels=3, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=32, out_channels=64, kernel_size=3),)

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=64, out_channels=128, kernel_size=3),)
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=128, out_channels=256, kernel_size=3),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=256, out_channels=512, kernel_size=3),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1),
            Conv(in_channels=512, out_channels=1024, kernel_size=3),
        )
        
        self.extract_features = extract_features if type(extract_features)==list else ['layer1', 'layer2', 'layer3', 'layer4']

        self.fmap_c = [{'layer1':64, 'layer2':128, 'layer3':256, 'layer4':1024}[l] for l in self.extract_features]
        self.stride = [{'layer1':4, 'layer2':8, 'layer3':16, 'layer4':32}[l] for l in self.extract_features]      

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_layers = [] 
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):
                continue
            layer = getattr(self, name)
            x = layer(x)
            if name in self.extract_features:
                out_layers.append(x)
        return out_layers 



class DarkNet19(nn.Module):
    """DarkNet19 <https://arxiv.org/pdf/1612.08242.pdf>"""

    def __init__(self, num_classes: int = 1000, init_weight: bool = True,extract_features:bool =True) -> None:
        super().__init__()

        if init_weight:
            self.apply(_initialize_weights)

        self.layer1 = nn.Sequential(
            Conv(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv(in_channels=64, out_channels=128, kernel_size=3),
            Conv(in_channels=128, out_channels=64, kernel_size=1),
            Conv(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv(in_channels=128, out_channels=256, kernel_size=3),
            Conv(in_channels=256, out_channels=128, kernel_size=1),
            Conv(in_channels=128, out_channels=256, kernel_size=3),

        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=256, out_channels=512, kernel_size=3),
            Conv(in_channels=512, out_channels=256, kernel_size=1),
            Conv(in_channels=256, out_channels=512, kernel_size=3),
            Conv(in_channels=512, out_channels=256, kernel_size=1),
            Conv(in_channels=256, out_channels=512, kernel_size=3),
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=512, out_channels=1024, kernel_size=3),
            Conv(in_channels=1024, out_channels=512, kernel_size=1),
            Conv(in_channels=512, out_channels=1024, kernel_size=3),
            Conv(in_channels=1024, out_channels=512, kernel_size=1),
            Conv(in_channels=512, out_channels=1024, kernel_size=3),
        )

        self.fmap_c = [256, 512, 1024]
        self.stride = [8, 16, 32]
        self.classifier = None
        if not extract_features:
            self.classifier = nn.Sequential(
                Conv(in_channels=1024, out_channels=num_classes, kernel_size=1),
                GlobalAvgPool2d()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        if self.classifier is not None:
            return self.classifier(f3)
        return [f1, f2, f3]


class DarkNet53(nn.Module):
    """DarkNet53 <https://pjreddie.com/media/files/papers/YOLOv3.pdf>"""

    def __init__(self, block: Callable[..., nn.Module], num_classes: int = 1000, init_weight: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        if init_weight:
            self.apply(_initialize_weights)

        self.features = nn.Sequential(
            Conv(in_channels=3, out_channels=32, kernel_size=3),

            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            *self._make_layer(block, in_channels=64, num_blocks=1),

            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            *self._make_layer(block, in_channels=128, num_blocks=2),

            Conv(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            *self._make_layer(block, in_channels=256, num_blocks=8),

            Conv(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            *self._make_layer(block, in_channels=512, num_blocks=8),

            Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
            *self._make_layer(block, in_channels=1024, num_blocks=4)
        )
        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    @staticmethod
    def _make_layer(block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


class CSPDarkNet53(nn.Module):
    """Cross Stage Partial DarkNet53 <https://pjreddie.com/media/files/papers/YOLOv3.pdf>"""

    def __init__(self, block: Callable[..., nn.Module], num_classes: int = 1000, init_weight: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        if init_weight:
            self.apply(_initialize_weights)

        self.features = nn.Sequential(
            Conv(in_channels=3, out_channels=32, kernel_size=3),

            Conv(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            block(64, 64, num_blocks=1),

            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            block(128, 128, num_blocks=2),

            Conv(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            block(256, 256, num_blocks=8),

            Conv(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            block(512, 512, num_blocks=8),

            Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
            block(1024, 1024, num_blocks=4)
        )
        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

def darknet7(num_classes: int = 1000, init_weight: bool = True,extract_features:bool =True) -> DarkNet7:
    return DarkNet7(num_classes=num_classes, init_weight=init_weight,extract_features=extract_features)

def darknet7f(num_classes: int = 1000, init_weight: bool = True,extract_features = ['layer1', 'layer2', 'layer3', 'layer4']) -> DarkNet7F:
    return DarkNet7F(num_classes=num_classes, init_weight=init_weight,extract_features=extract_features)

def darknet19(num_classes: int = 1000, init_weight: bool = True,extract_features:bool =True) -> DarkNet19:
    return DarkNet19(num_classes=num_classes, init_weight=init_weight,extract_features =extract_features)


def darknet53(num_classes: int = 1000, init_weight: bool = True) -> DarkNet53:
    return DarkNet53(ResidualBlock, num_classes=num_classes, init_weight=init_weight)


def darknet53e(num_classes: int = 1000, init_weight: bool = True) -> DarkNet53:
    """DarkNet53 with ELASTIC block"""
    return DarkNet53(Elastic, num_classes=num_classes, init_weight=init_weight)


def cspdarknet53(num_classes: int = 1000, init_weight: bool = True) -> CSPDarkNet53:
    """DarkNet53 with CSP block"""
    return CSPDarkNet53(CSP, num_classes=num_classes, init_weight=init_weight)


def num_params(model: Callable[..., nn.Module]) -> int:
    _num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _num

@BACKBONES.register_module
class DarknetWrapper(nn.Module):

    def __init__(self,
                darknet = 'darknet7',
                num_classes: int = 1000,
                init_weight: bool = True,
                extract_features = True,
                out_conv = False,
                cfg=None):
        super(DarknetWrapper, self).__init__()
        self.cfg = cfg

        self.model = eval(darknet)(num_classes, init_weight, extract_features)

        if darknet=='darknet7f':
            load_darknet_weights(self.model, '/root/data1/ws/SNN_CV/pretrained/yolov3-tiny.weights')

        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = nn.Conv2d(self.model.fmap_c[-1], cfg.featuremap_out_channel,kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    # print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
    # by ysyun, use .view_as()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias.data));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data));
    start = start + num_w
    return start

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    # conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data));
    start = start + num_w
    return start


def load_darknet_weights(model, weights_path):
    print(f'start load pretrained weight from: {weights_path} ...')
    fp = open(weights_path, 'rb')

    version = np.fromfile(fp, count=3, dtype=np.int32)
    version = [int(i) for i in version]
    if version[0] * 10 + version[1] >= 2 and version[0] < 1000 and version[1] < 1000:
        seen = np.fromfile(fp, count=1, dtype=np.int64)
    else:
        seen = np.fromfile(fp, count=1, dtype=np.int32)
    header = torch.from_numpy(np.concatenate((version, seen), axis=0))
    seen = int(seen)
    buf = np.fromfile(fp, dtype=np.float32)
    fp.close()
    

    start = 0
    ind = -2
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, name)
        for block in layer :
            # block = int(block['batch_normalize'])
            # print(type(layer) == nn.Conv2d)
            if type(block) == Conv:
                start = load_conv_bn(buf, start, block.conv, block.norm)
           
    
if __name__ == '__main__':
    net = DarknetWrapper()
    x=torch.rand((1,3,64,64))
    y=net(x)
    print(y)