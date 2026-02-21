import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Callable

from detection.models.registry import BACKBONES

from .darknet import GlobalAvgPool2d, _initialize_weights

from snn.spiking_wrappers import  SpikConv
from spikingjelly.clock_driven import surrogate
from snn.neuro import get_neuro

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, cnf: str = None, norm: dict = {'type':'BN'},neuro: dict= {'type':'IF'}):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpikConv(in_channels=inplanes, out_channels=planes, kernel_size=3,stride=stride, padding=dilation, dilation=dilation,
                              norm=norm,neuro=neuro)
        self.conv2 = SpikConv(in_channels=planes, out_channels=planes, kernel_size=3, padding=dilation, dilation=dilation,
                              norm=norm,neuro=neuro)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = get_neuro(neuro)
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
       
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        out = sew_function(identity, out, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, cnf: str = None, norm: dict = {'type':'BN'},neuro: dict= {'type':'IF'}):
        super(Bottleneck, self).__init__()
           
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpikConv(in_channels=inplanes, out_channels=width, kernel_size=1,norm=norm,neuro=neuro)

        self.conv2 = SpikConv(in_channels=width, out_channels=width, kernel_size=3,stride=stride,
                              groups=groups, dilation=dilation,norm=norm,neuro=neuro)

        self.conv3 = SpikConv(in_channels=width, out_channels=planes * self.expansion, kernel_size=1,norm=norm,neuro=neuro)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = get_neuro(neuro)
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        
        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(out, identity, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'

@BACKBONES.register_module
class SewResnet(nn.Module):
    def __init__(self,
                arch = 'resnet18',
                in_channels=3,
                zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                cnf: str = 'ADD',
                norm: dict = {'type':'BN'},
                neuro={'type':'IFNode'},
                out_features=['layer1', 'layer2', 'layer3', 'layer4'],
                num_classes=0,
                has_embeding=False,
                cfg=None):
        super(SewResnet, self).__init__()
        self.cfg = cfg
        self.cnf=cnf

        block={'resnet18':BasicBlock,
               'resnet34':BasicBlock,
               'resnet50':Bottleneck,
               'resnet101':Bottleneck,
               'resnet152':Bottleneck,}[arch]
        
        layers={'resnet18':[2, 2, 2, 2],
               'resnet34':[3, 4, 6, 3],
               'resnet50':[3, 4, 6, 3],
               'resnet101':[3, 4, 23, 3],
               'resnet152':[3, 8, 36, 3],}[arch]
        

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = SpikConv(in_channels=in_channels, out_channels=self.inplanes, 
                              kernel_size=7,stride=2,padding=3,norm=norm,neuro=neuro)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, layer1_chn,layer1_strd = self._make_layer(block, 64, layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, layer2_chn,layer2_strd = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, layer3_chn,layer3_strd  = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, layer4_chn,layer4_strd  = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
        
        self.chn_str={'layer1':[layer1_chn,layer1_strd*4],
                    'layer2':[layer2_chn,layer1_strd*layer2_strd*4],
                    'layer3':[layer3_chn,layer1_strd*layer2_strd*layer3_strd*4],
                    'layer4':[layer4_chn,layer1_strd*layer2_strd*layer3_strd*layer4_strd*4]}

        self.out_features=out_features

        self.has_embeding=has_embeding
        if has_embeding:
            # self.feature_dconv = torch.nn.ModuleList(
            #     [ conv1x1(self.chn_str[name][0], self.chn_str[name][0], stride=1) for name in self.out_features])
            self.feature_dconv = torch.nn.ModuleList(
                [ SpikConv(in_channels=self.chn_str[name][0], out_channels=self.chn_str[name][0], kernel_size=1,norm=norm,neuro=neuro) for name in self.out_features]
                )

        if num_classes>0:
            self.out_features=[]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str=None, norm: dict = {'type':'BN'}, neuro={'type':'IF'},):
        norm_layer = nn.BatchNorm2d if norm['type']=='BN' else None
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, cnf, norm=norm,neuro=neuro))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                cnf=cnf, norm=norm,neuro=neuro))

        return nn.Sequential(*layers), planes, stride
    
    def embeding(self, features):
        self.embeding_features={}
        if not self.has_embeding:
            return 
        assert len(features)>=len(self.out_features)
        emb_features=features[-len(self.out_features):]

        for i,n in enumerate(self.out_features):
            f=emb_features[i]
            dconv=self.feature_dconv[i]
            self.embeding_features[n] = dconv(f)
            # dconv(f) for f, dconv in zip(emb_features, self.feature_dconv)

    def _forward_ss(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        out_layers = [] 
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):
                continue
            layer = getattr(self, name)
            x = layer(x)
            if name in self.out_features:
                out_layers.append(x)

        if hasattr(self,'fc'):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
        return out_layers
    
    def init_out_spike(self,B,H,W,T,device):
        outputs={}
        for n in self.out_features:
            chn,stride=self.chn_str[n]
            nW = W//stride
            nH = H//stride
            outputs[n]=torch.zeros(B, chn, nH, nW, T, device=device)
        return outputs
    
    def get_zeroTS_output(self,xs):
        # B,_,H,W,TS = xs.shape
        if xs.dim()==5:
            B,_,H,W,TS = xs.shape
        else:
            B,_,H,W = xs.shape
            TS = 10
        outputs_dict = self.init_out_spike(B,H,W,TS,xs.device)

        return [outputs_dict[n] for n in self.out_features]
    
    def _forward_ms(self, xs):
        B,_,H,W,TS = xs.shape

        outputs_dict = self.init_out_spike(B,H,W,TS,xs.device)
        # outputs_dict={}

        for t in range(TS):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)
                if self.has_embeding:
                    if name in self.embeding_features.keys() and t==0:
                        x = sew_function(x, self.embeding_features[name], self.cnf)
                if name in self.out_features:
                    outputs_dict[name][...,t]=x
                    # outputs_dict[name]=x
        
        out_layers=[outputs_dict[n] for n in self.out_features]

        if hasattr(self,'fc'):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
        return out_layers

    def forward(self, encode_x):
        if encode_x.dim() == 5 and self.maxpool.__class__.__name__[:2]!='ms':
            return self._forward_ms(encode_x)
        else:
            return self._forward_ss(encode_x)


@BACKBONES.register_module
class SewResnetI(nn.Module):
    def __init__(self,
                arch = 'resnet18',
                in_channels=3,
                zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                cnf: str = 'ADD',
                norm: dict = {'type':'BN'},
                neuro={'type':'IFNode'},
                out_features=['layer1', 'layer2', 'layer3', 'layer4'],
                num_classes=0,
                cfg=None):
        super(SewResnetI, self).__init__()
        self.cfg = cfg
        self.cnf=cnf

        block={'resnet18':BasicBlock,
               'resnet34':BasicBlock,
               'resnet50':Bottleneck,
               'resnet101':Bottleneck,
               'resnet152':Bottleneck,}[arch]
        
        layers={'resnet18':[2, 2, 2, 2],
               'resnet34':[3, 4, 6, 3],
               'resnet50':[3, 4, 6, 3],
               'resnet101':[3, 4, 23, 3],
               'resnet152':[3, 8, 36, 3],}[arch]
        

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = SpikConv(in_channels=in_channels, out_channels=self.inplanes, 
                              kernel_size=7,stride=2,padding=3,norm=norm,neuro=neuro)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, layer1_chn,layer1_strd = self._make_layer(block, 64, layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, layer2_chn,layer2_strd = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, layer3_chn,layer3_strd  = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, layer4_chn,layer4_strd  = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
        
        self.chn_str={  'layer1':[layer1_chn,layer1_strd*4],
                        'layer2':[layer2_chn,layer1_strd*layer2_strd*4],
                        'layer3':[layer3_chn,layer1_strd*layer2_strd*layer3_strd*4],
                        'layer4':[layer4_chn,layer1_strd*layer2_strd*layer3_strd*layer4_strd*4]}

        self.out_features=out_features

        if num_classes>0:
            self.out_features=[]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str=None, norm: dict = {'type':'BN'}, neuro={'type':'IF'},):
        norm_layer = nn.BatchNorm2d if norm['type']=='BN' else None
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, cnf, norm=norm,neuro=neuro))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                cnf=cnf, norm=norm,neuro=neuro))

        return nn.Sequential(*layers), planes, stride
        
    def init_out_spike(self,B,H,W,T,device):
        outputs={}
        for n in self.out_features:
            chn,stride=self.chn_str[n]
            nW = W//stride
            nH = H//stride
            outputs[n]=torch.zeros(B, chn, nH, nW, T, device=device)
        return outputs
    
    def forward(self, xs, **kwargs):
        B,_,H,W,TS = xs.shape

        outputs_dict = self.init_out_spike(B,H,W,TS+1,xs.device)
        # outputs_dict={}
        for t in range(TS):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                if name in self.out_features:
                    outputs_dict[name][...,t+1]=x
        
        out_layers=[outputs_dict[n] for n in self.out_features]

        if hasattr(self,'fc'):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
        return out_layers


# if __name__ == '__main__':
#     net = DarknetWrapper()
#     x=torch.rand((1,3,64,64))
#     y=net(x)
#     print(y)