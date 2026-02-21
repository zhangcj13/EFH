import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Callable, Tuple
from detection.models.registry import BACKBONES

from .darknet import GlobalAvgPool2d, _initialize_weights

from snn.spiking_wrappers import  SpikConv
from spikingjelly.clock_driven import surrogate
from snn.neuro import get_neuro

from .sew_resnet import BasicBlock,Bottleneck,sew_function,conv1x1,conv3x3
from spikingjelly.clock_driven.rnn import SpikingLSTM, SpikingVanillaRNN, SpikingGRU

pretrained_weight_path = {
    'resnet18': '/root/data1/ws/SNN_CV/pretrained/sew18_checkpoint_319.pth',
    'resnet34': '/root/data1/ws/SNN_CV/pretrained/sew34_checkpoint_319.pth',
}

class FeatureFusion(nn.Module):
    def __init__(self, channel,norm=False):
        super(FeatureFusion, self).__init__()
        self.norm = norm
        self.fusion = self.build_layer(channel)

    def build_layer(self, channel):
        if not self.norm:
            return nn.Sequential(
                nn.Conv2d(channel*2, int(channel/2), kernel_size=1, stride=1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/2), int(channel/4), kernel_size=7, padding=3), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), int(channel/4), kernel_size=7, padding=3), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), channel, kernel_size=1),nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(channel*2, int(channel/2), kernel_size=1, stride=1), nn.BatchNorm2d(int(channel/2)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/2), int(channel/4), kernel_size=7, padding=3), nn.BatchNorm2d(int(channel/4)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), int(channel/4), kernel_size=7, padding=3), nn.BatchNorm2d(int(channel/4)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), channel, kernel_size=1), nn.BatchNorm2d(channel),nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, image_fea, event_fea):
        if event_fea is None:
            event_fea = torch.zeros_like(image_fea)
        x = torch.cat([image_fea, event_fea], dim=1)
        res = self.fusion(x)
        return res

class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: torch.Tensor, h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)
        xh = torch.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t

class SpikDWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.,
                 surrogate_function1=surrogate.Erf(), 
                 surrogate_function2=None):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        
        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: torch.Tensor, h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)
            h_tm1 = self.surrogate_function1(h_tm1)
        xh = torch.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
            xh = self.surrogate_function1(h_tm1)
        mix = self.conv1x1(xh)


        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        gates = self.surrogate_function1(gates)
        if self.surrogate_function2 is None:
            cell_input = self.surrogate_function1(cell_input)
        else:
            cell_input = self.surrogate_function2(cell_input)

        assert gates.shape[1] == cell_input.shape[1] * 3

        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(cell_input)

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * c_t

        return h_t, c_t

@BACKBONES.register_module
class SewResnetRNN(nn.Module):
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
                out_type='last',
                num_classes=0,
                has_embeding=False,
                rnn_type = 'lstm',
                cfg=None):
        super(SewResnetRNN, self).__init__()
        self.cfg = cfg
        self.cnf=cnf
        self.out_type = out_type

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
            # self.feature_dconv ={}
            # self.feature_dconv_name =[]
            for k in self.chn_str.keys():
                setattr(self,'feature_dconv_'+k,SpikConv(in_channels=self.chn_str[k][0], out_channels=self.chn_str[k][0], kernel_size=1,norm=norm,neuro=neuro))
                # self.feature_dconv[k]=SpikConv(in_channels=self.chn_str[k][0], out_channels=self.chn_str[k][0], kernel_size=1,norm=norm,neuro=neuro)
                
        
        if rnn_type == 'lstm':
            # self.rnn_blocks={}
            for k in self.chn_str.keys():
                setattr(self,'rnn_'+k,SpikDWSConvLSTM2d(dim = self.chn_str[k][0],
                                                       dws_conv = True,
                                                       dws_conv_only_hidden = True,
                                                       dws_conv_kernel_size = 3,
                                                       cell_update_dropout = 0.))
                # self.rnn_blocks[k] = SpikDWSConvLSTM2d(dim = self.chn_str[k][0],
                #                                        dws_conv = True,
                #                                        dws_conv_only_hidden = True,
                #                                        dws_conv_kernel_size = 3,
                #                                        cell_update_dropout = 0.)
                


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
        self.h_and_c_previous = {k:None for k in self.chn_str.keys()}

        self.embeding_features={}
        if not self.has_embeding:
            return 
        len_dconv = len(self.chn_str.keys())
        assert len(features)>=len_dconv
        emb_features=features[-len_dconv:]

        # for i,n in enumerate(self.chn_str.keys()):
        #     f=emb_features[i]
        #     dconv = getattr(self,'feature_dconv_'+n)
        #     self.embeding_features[n] = dconv(f)
        
        for i,n in enumerate(self.chn_str.keys()):
            f=emb_features[i]
            dconv = getattr(self,'feature_dconv_'+n)
            rnn_block = getattr(self,'rnn_'+n)
            x = dconv(f)
            h_c, c_c = rnn_block(x, self.h_and_c_previous[n])
            self.h_and_c_previous[n] = [h_c,c_c]
            

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
    
    def _forward_ms(self, xs):
        B,_,H,W,TS = xs.shape

        if self.out_type == 'last':
            outputs_dict = {}
        else:
            outputs_dict = self.init_out_spike(B,H,W,TS,xs.device)
        # outputs_dict={}
        if not hasattr(self,'h_and_c_previous'):
            self.h_and_c_previous = {k:None for k in self.chn_str.keys()}

        for t in range(TS):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                if name in self.chn_str.keys():
                    rnn_block = getattr(self,'rnn_'+name)
                    h_c,c_c = rnn_block(x,  self.h_and_c_previous[name])
                    self.h_and_c_previous[name] = [h_c,c_c]
                    x = h_c

                if name in self.out_features:
                    if  self.out_type == 'last':
                        if t == TS - 1:
                            outputs_dict[name] = x
                    else:
                        outputs_dict[name][...,t]=x

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
class SewResnetRNNv2(nn.Module):
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
                out_type='last',
                num_classes=0,
                has_embeding=False,
                rnn_type = 'lstm',
                cfg=None):
        super(SewResnetRNNv2, self).__init__()
        self.cfg = cfg
        self.cnf=cnf
        self.out_type = out_type

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

        for k in self.chn_str.keys():
            setattr(self,'feature_dconv_'+k,SpikConv(in_channels=self.chn_str[k][0], out_channels=self.chn_str[k][0], kernel_size=1,norm=norm,neuro=neuro))
            
        
  
        for k in self.chn_str.keys():
            setattr(self,'rnn_'+k,SpikDWSConvLSTM2d(dim = self.chn_str[k][0],
                                                    dws_conv = True,
                                                    dws_conv_only_hidden = True,
                                                    dws_conv_kernel_size = 3,
                                                    cell_update_dropout = 0.))
        for k in self.chn_str.keys():
            if k !='layer1':
                setattr(self,'feature_fusion_'+ k, FeatureFusion(channel = self.chn_str[k][0],
                                                                 norm=True,
                                                                 ))
                setattr(self,'feature_rnn_'+ k, DWSConvLSTM2d(dim = self.chn_str[k][0],
                                                        dws_conv = True,
                                                        dws_conv_only_hidden = True,
                                                        dws_conv_kernel_size = 3,
                                                        cell_update_dropout = 0.))
                    
                


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
        self.h_and_c_previous = {k:None for k in self.chn_str.keys()}
        self.feature_h_and_c_previous = {k:None for k in self.chn_str.keys()}

        self.embeding_features={}
        if not self.has_embeding:
            return 
        len_dconv = len(self.chn_str.keys())
        assert len(features)>=len_dconv
        emb_features=features[-len_dconv:]
        
        for i,n in enumerate(self.chn_str.keys()):
            f=emb_features[i]
            dconv = getattr(self,'feature_dconv_'+n)
            rnn_block = getattr(self,'rnn_'+n)
            x = dconv(f)
            h_c, c_c = rnn_block(x, self.h_and_c_previous[n])
            self.h_and_c_previous[n] = [h_c,c_c]
        
        for i,n in enumerate(self.chn_str.keys()):
            if i==0:
                continue
            f=emb_features[i]
            fusion_block = getattr(self,'feature_fusion_'+n)
            feature_rnn_block = getattr(self,'feature_rnn_'+n)
            y = fusion_block(f, None)
            fh_c, fc_c = feature_rnn_block(y, self.feature_h_and_c_previous[n])
            self.feature_h_and_c_previous[n] = [fh_c,fc_c]
            
    
    def init_out_spike(self,B,H,W,T,device):
        outputs={}
        for n in self.out_features:
            chn,stride=self.chn_str[n]
            nW = W//stride
            nH = H//stride
            outputs[n]=torch.zeros(B, chn, nH, nW, T, device=device)
        return outputs
    
    def _forward_ms(self, xs):
        if xs is None:
            return [self.feature_h_and_c_previous[name][0] for name in self.out_features]
                

        B,_,H,W,TS = xs.shape

        if self.out_type == 'last':
            outputs_dict = {}
        else:
            outputs_dict = self.init_out_spike(B,H,W,TS,xs.device)
        # outputs_dict={}
        if not hasattr(self,'h_and_c_previous'):
            self.h_and_c_previous = {k:None for k in self.chn_str.keys()}

        for t in range(TS):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                if name in self.chn_str.keys():
                    rnn_block = getattr(self,'rnn_'+name)
                    h_c,c_c = rnn_block(x,  self.h_and_c_previous[name])
                    self.h_and_c_previous[name] = [h_c,c_c]
                    x = h_c

                if name in self.out_features:
                    fusion_block = getattr(self,'feature_fusion_'+name)
                    feature_rnn_block = getattr(self,'feature_rnn_'+name)
                    fh_c,_ = self.feature_h_and_c_previous[name]
                    y = fusion_block(fh_c, x)
                    fh_c, fc_c = feature_rnn_block(y,  self.feature_h_and_c_previous[name])
                    self.feature_h_and_c_previous[name] = [fh_c,fc_c]
                    
                    if t == TS - 1:
                        outputs_dict[name] = fh_c


        out_layers=[outputs_dict[n] for n in self.out_features]

        if hasattr(self,'fc'):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
        return out_layers

    def forward(self, encode_x):
        return self._forward_ms(encode_x)


@BACKBONES.register_module
class SewResnetDRNN(nn.Module):
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
                cfg=None):
        super(SewResnetDRNN, self).__init__()
        self.cfg = cfg
        self.cnf=cnf

        block={'resnet10':BasicBlock,
               'resnet18':BasicBlock,
               'resnet34':BasicBlock,
               'resnet50':Bottleneck,
               'resnet101':Bottleneck,
               'resnet152':Bottleneck,}[arch]
        
        layers={'resnet10':[1, 1, 1, 1],
                'resnet18':[2, 2, 2, 2],
                'resnet34':[3, 4, 6, 3],
                'resnet50':[3, 4, 6, 3],
                'resnet101':[3, 4, 23, 3],
                'resnet152':[3, 8, 36, 3],}[arch]
        
        ichn={'resnet10':[64, 64, 128, 256],
              'resnet18':[64, 128, 256, 512],
              'resnet34':[64, 128, 256, 512],
              'resnet50':[64, 128, 256, 512],
              'resnet101':[64, 128, 256, 512],
              'resnet152':[64, 128, 256, 512]}[arch]
        
        self.ichn=ichn
        
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

        self.layer1, _,_ = self._make_layer(block, ichn[0], layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, _,_ = self._make_layer(block, ichn[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, _,_  = self._make_layer(block, ichn[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, _,_  = self._make_layer(block, ichn[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
    
        self.out_features=out_features

        self.expansion = block.expansion

        self.emd_layer1=SpikConv(ichn[0]*self.expansion, ichn[0]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer2=SpikConv(ichn[1]*self.expansion, ichn[1]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer3=SpikConv(ichn[2]*self.expansion, ichn[2]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer4=SpikConv(ichn[3]*self.expansion, ichn[3]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

  
        self.rnn_layer1=DWSConvLSTM2d(dim = ichn[0]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer2=DWSConvLSTM2d(dim = ichn[1]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer3=DWSConvLSTM2d(dim = ichn[2]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer4=DWSConvLSTM2d(dim = ichn[3]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)


        self.snh_layer1=SpikConv(ichn[0]*self.expansion, ichn[0]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.snh_layer2=SpikConv(ichn[1]*self.expansion, ichn[1]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.snh_layer3=SpikConv(ichn[2]*self.expansion, ichn[2]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        # self.snh_layer4=SpikConv(512*self.expansion, 512*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

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

    def init_rnn_output(self,B,H,W,T,device):
        outputs={'layer1':torch.zeros(B,  64*self.expansion,  H//4,  W//4, T, device=device), 
                 'layer2':torch.zeros(B, 128*self.expansion,  H//8,  W//8, T, device=device), 
                 'layer3':torch.zeros(B, 256*self.expansion, H//16, W//16, T, device=device), 
                 'layer4':torch.zeros(B, 512*self.expansion, H//32, W//32, T, device=device), 
                }
        return outputs
    
    def forward(self, xs, features=None, max_t=None):
        B,_,H,W,TS = xs.shape
        outputs_dict = self.init_rnn_output(B,H,W,TS+1,xs.device)
        self.h_and_c_previous = {k:None for k in ['layer1', 'layer2', 'layer3', 'layer4']}

        if features is not None:
            assert len(features)>=4, 'length of features must greater than 3'
            features = features[-4:]
            for i,name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
                emd_block = getattr(self,'emd_'+name)
                rnn_block = getattr(self,'rnn_'+name)
                fea = features[i]
                fea = emd_block(fea)
                h_c,c_c = rnn_block(fea,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                outputs_dict[name][...,0]=h_c
        
        if max_t is None:
            max_t = TS 

        for t in range(max_t):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                rnn_block = getattr(self,'rnn_'+name)
                h_c,c_c = rnn_block(x,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                x = h_c

                outputs_dict[name][...,t+1]=x

                if hasattr(self, 'snh_'+name):
                    snh_block = getattr(self,'snh_'+name)
                    x = snh_block(x)

        out_layers = [outputs_dict[n] for n in self.out_features]
        return out_layers

@BACKBONES.register_module
class SewResnetSDRNN(nn.Module):
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
                cfg=None):
        super(SewResnetSDRNN, self).__init__()
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

        self.layer1, _,_ = self._make_layer(block, 64, layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, _,_ = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, _,_  = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, _,_  = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
    
        self.out_features=out_features

        self.expansion = block.expansion

        self.emd_layer1=SpikConv( 64*self.expansion,  64*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer2=SpikConv(128*self.expansion, 128*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer3=SpikConv(256*self.expansion, 256*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer4=SpikConv(512*self.expansion, 512*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

  
        self.rnn_layer1=SpikDWSConvLSTM2d(dim = 64*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer2=SpikDWSConvLSTM2d(dim = 128*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer3=SpikDWSConvLSTM2d(dim = 256*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer4=SpikDWSConvLSTM2d(dim = 512*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)


        self.snh_layer1=SpikConv( 64*self.expansion,  64*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.snh_layer2=SpikConv(128*self.expansion, 128*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.snh_layer3=SpikConv(256*self.expansion, 256*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        # self.snh_layer4=SpikConv(512*self.expansion, 512*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

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

    def init_rnn_output(self,B,H,W,T,device):
        outputs={'layer1':torch.zeros(B,  64*self.expansion,  H//4,  W//4, T, device=device), 
                 'layer2':torch.zeros(B, 128*self.expansion,  H//8,  W//8, T, device=device), 
                 'layer3':torch.zeros(B, 256*self.expansion, H//16, W//16, T, device=device), 
                 'layer4':torch.zeros(B, 512*self.expansion, H//32, W//32, T, device=device), 
                }
        return outputs
    
    def forward(self, xs, features=None, max_t=None):
        B,_,H,W,TS = xs.shape
        outputs_dict = self.init_rnn_output(B,H,W,TS+1,xs.device)
        self.h_and_c_previous = {k:None for k in ['layer1', 'layer2', 'layer3', 'layer4']}

        if features is not None:
            assert len(features)>=4, 'length of features must greater than 3'
            features = features[-4:]
            for i,name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
                emd_block = getattr(self,'emd_'+name)
                rnn_block = getattr(self,'rnn_'+name)
                fea = features[i]
                fea = emd_block(fea)
                h_c,c_c = rnn_block(fea,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                outputs_dict[name][...,0]=h_c
        
        if max_t is None:
            max_t = TS 

        for t in range(max_t):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                rnn_block = getattr(self,'rnn_'+name)
                h_c,c_c = rnn_block(x,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                x = h_c

                outputs_dict[name][...,t+1]=x

                if hasattr(self, 'snh_'+name):
                    snh_block = getattr(self,'snh_'+name)
                    x = snh_block(x)

        out_layers = [outputs_dict[n] for n in self.out_features]
        return out_layers

@BACKBONES.register_module
class SewResnetSDRNN2(nn.Module):
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
                cfg=None):
        super(SewResnetSDRNN2, self).__init__()
        self.cfg = cfg
        self.cnf=cnf
        self.arch=arch
        block={'resnet10':BasicBlock,
               'resnet18':BasicBlock,
               'resnet18t':BasicBlock,
               'resnet18x':BasicBlock,
               'resnet34':BasicBlock,
               'resnet34t':BasicBlock,
               'resnet34x':BasicBlock,
               'resnet50':Bottleneck,
               'resnet101':Bottleneck,
               'resnet152':Bottleneck,}[arch]
        
        layers={'resnet10':[1, 1, 1, 1],
               'resnet18':[2, 2, 2, 2],
               'resnet18t':[2, 2, 2, 2],
               'resnet18x':[2, 2, 2, 2],
               'resnet34':[3, 4, 6, 3],
               'resnet34t':[3, 4, 6, 3],
               'resnet34x':[3, 4, 6, 3],
               'resnet50':[3, 4, 6, 3],
               'resnet101':[3, 4, 23, 3],
               'resnet152':[3, 8, 36, 3],}[arch]
        
        ichn={'resnet10':[64, 64, 128, 256],
              'resnet18':[64, 128, 256, 512],
              'resnet18t':[64, 64, 128, 256],
              'resnet18x':[64, 128, 256, 360],
              'resnet34x':[64, 128, 256, 360],
              'resnet34':[64, 128, 256, 512],
              'resnet34t':[64, 64, 128, 256],
              'resnet50':[64, 128, 256, 512],
              'resnet101':[64, 128, 256, 512],
              'resnet152':[64, 128, 256, 512]}[arch]
        self.ichn=ichn
        
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

        self.layer1, layer1_chn,layer1_strd = self._make_layer(block, ichn[0], layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, layer2_chn,layer2_strd  = self._make_layer(block, ichn[1], layers[1], stride=2,
                                                                dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, layer3_chn,layer3_strd  = self._make_layer(block, ichn[2], layers[2], stride=2,
                                                                dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, layer4_chn,layer4_strd  = self._make_layer(block, ichn[3], layers[3], stride=2,
                                                                dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
    
        self.chn_str={  'layer1':[layer1_chn,layer1_strd*4],
                        'layer2':[layer2_chn,layer1_strd*layer2_strd*4],
                        'layer3':[layer3_chn,layer1_strd*layer2_strd*layer3_strd*4],
                        'layer4':[layer4_chn,layer1_strd*layer2_strd*layer3_strd*layer4_strd*4]}
        
        self.out_features=out_features

        self.expansion = block.expansion

        self.emd_layer1=SpikConv(ichn[0]*self.expansion, ichn[0]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer2=SpikConv(ichn[1]*self.expansion, ichn[1]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer3=SpikConv(ichn[2]*self.expansion, ichn[2]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer4=SpikConv(ichn[3]*self.expansion, ichn[3]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

  
        self.rnn_layer1=SpikDWSConvLSTM2d(dim = ichn[0]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer2=SpikDWSConvLSTM2d(dim = ichn[1]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer3=SpikDWSConvLSTM2d(dim = ichn[2]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)
        self.rnn_layer4=SpikDWSConvLSTM2d(dim = ichn[3]*self.expansion, dws_conv = True, dws_conv_only_hidden = True, dws_conv_kernel_size = 3,cell_update_dropout = 0.)

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

    def init_rnn_output(self,B,H,W,T,device):
        # outputs={'layer1':torch.zeros(B, self.ichn[0],  H//4,  W//4, T, device=device), 
        #          'layer2':torch.zeros(B, self.ichn[1],  H//8,  W//8, T, device=device), 
        #          'layer3':torch.zeros(B, self.ichn[2], H//16, W//16, T, device=device), 
        #          'layer4':torch.zeros(B, self.ichn[3], H//32, W//32, T, device=device), 
        #         }
        outputs={}
        for i,name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            _, stride=self.chn_str[name]
            # outputs[name]=torch.zeros(B, self.ichn[i], H//stride, W//stride, T, device=device)
            outputs[name]=torch.zeros(B, self.ichn[i], round(H/stride), round(W/stride), T, device=device)

        
        return outputs
    
    def forward(self, xs, features=None, max_t=None):
        B,_,H,W,TS = xs.shape
        outputs_dict = self.init_rnn_output(B,H,W,TS+1,xs.device)
        self.h_and_c_previous = {k:None for k in ['layer1', 'layer2', 'layer3', 'layer4']}

        if features is not None:
            assert len(features)>=4, 'length of features must greater than 3'
            features = features[-4:]
            for i,name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
                emd_block = getattr(self,'emd_'+name)
                rnn_block = getattr(self,'rnn_'+name)
                fea = features[i]
                fea = emd_block(fea)
                h_c,c_c = rnn_block(fea,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                outputs_dict[name][...,0]=h_c
        
        if max_t is None:
            max_t = TS 

        for t in range(max_t):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                rnn_block = getattr(self,'rnn_'+name)
                h_c,c_c = rnn_block(x,  self.h_and_c_previous[name])
                self.h_and_c_previous[name] = [h_c,c_c]
                x = h_c

                outputs_dict[name][...,t+1]=x

                # if hasattr(self, 'snh_'+name):
                #     snh_block = getattr(self,'snh_'+name)
                #     x = snh_block(x)

        out_layers = [outputs_dict[n] for n in self.out_features]
        return out_layers

    def load_pretrained(self,):
        from pretrained.match_keys import sewresnet34_keys

        if self.arch not in pretrained_weight_path.keys():
            print(f'---------- no weight for arch: {self.arch} ----------')
            return
        
        weight_path = pretrained_weight_path[self.arch]
                
        match_keys = {'resnet34':sewresnet34_keys}[self.arch]

        ckpt_info = torch.load(weight_path)
        
        pretrained_net = ckpt_info['model']
        net_state = self.state_dict()
        state = {}
        
        for k, v in pretrained_net.items():
            nk = ''
            if k in match_keys.keys():
                nk= match_keys[k]            
            
            if nk not in net_state.keys() or v.size() != net_state[nk].size():
                print('skip weights: ' + k)
                continue
            state[k] = v
        self.load_state_dict(state, strict=False)

        print(f'********** load pretrained from: {weight_path}:')

    
    def exp_pretrained_keys(self,):
        if self.arch not in pretrained_weight_path.keys():
            print(f'---------- no weight for arch: {self.arch} ----------')
            return
        
        weight_path = pretrained_weight_path[self.arch]
        print(f'********** load pretrained from: {weight_path}:')

        ckpt_info = torch.load(weight_path)
        
        pretrained_net = ckpt_info['model']
        net_state = self.state_dict()
        state = {}
        match_keys={}
        # for k, v in net_state.items():
        #     if k[:6]=='layer2':
        #         print(f'{k}')
        print(f'-----------------------------------')
        cnt=0
        for k, v in pretrained_net.items():
            if k[:5]=='conv1':
                # print(f'{k}')
                match_keys[k]='conv1.conv.weight'
            elif k[:3]=='bn1':
                match_keys[k]='conv1.norm.'+k[4:]
            elif k[:5]=='layer':
                # print(f'{k}')
                seps= k.split('.')

                if 'downsample' in k:
                    nk=f'{seps[0]}.{seps[1]}.{seps[2]}.{seps[5]}.{seps[6]}'
                else:
                    tk={'0':'conv',
                        '1':'norm'}[seps[4]]
                    nk=f'{seps[0]}.{seps[1]}.{seps[2]}.{tk}.{seps[5]}'
                
                # match_keys[nk]=k
                match_keys[k]=nk
                # print(f'{nk} : {k}')
            else:
                print(f'{k}')
            cnt+=1
        #     if k not in net_state.keys() or v.size() != net_state[k].size():
        #         print('skip weights: ' + k)
        #         continue
        #     state[k] = v
        # self.load_state_dict(state, strict=False)
        print(f'-----------------------------------')
        for k, v in match_keys.items():
            print(f'\'{k}\': \'{v}\',')
        # print(match_keys, len(match_keys.keys()),'/',cnt,'/',len(pretrained_net.keys()))

@BACKBONES.register_module
class SewResnetNRNN(nn.Module):
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
                cfg=None):
        super(SewResnetNRNN, self).__init__()
        self.cfg = cfg
        self.cnf=cnf
        
        block={'resnet10':BasicBlock,
               'resnet18':BasicBlock,
               'resnet18t':BasicBlock,
               'resnet34':BasicBlock,
               'resnet34t':BasicBlock,
               'resnet50':Bottleneck,
               'resnet101':Bottleneck,
               'resnet152':Bottleneck,}[arch]
               
        layers={'resnet10':[1, 1, 1, 1],
               'resnet18':[2, 2, 2, 2],
               'resnet18t':[2, 2, 2, 2],
               'resnet34':[3, 4, 6, 3],
               'resnet34t':[3, 4, 6, 3],
               'resnet50':[3, 4, 6, 3],
               'resnet101':[3, 4, 23, 3],
               'resnet152':[3, 8, 36, 3],}[arch]
        
        ichn={'resnet10':[64, 64, 128, 256],
              'resnet18':[64, 128, 256, 512],
              'resnet18t':[64, 64, 128, 256],
              'resnet34':[64, 128, 256, 512],
              'resnet34t':[64, 64, 128, 256],
              'resnet50':[64, 128, 256, 512],
              'resnet101':[64, 128, 256, 512],
              'resnet152':[64, 128, 256, 512]}[arch]
        
        self.ichn=ichn

        

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

        self.layer1, _,_ = self._make_layer(block, ichn[0], layers[0], cnf=cnf,  norm=norm,neuro=neuro)
        self.layer2, _,_ = self._make_layer(block, ichn[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, norm=norm,neuro=neuro)
        self.layer3, _,_  = self._make_layer(block, ichn[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, norm=norm,neuro=neuro)
        self.layer4, _,_  = self._make_layer(block, ichn[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, norm=norm,neuro=neuro)
    
        self.out_features=out_features

        self.expansion = block.expansion

        self.emd_layer1=SpikConv(ichn[0]*self.expansion, ichn[0]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer2=SpikConv(ichn[1]*self.expansion, ichn[1]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer3=SpikConv(ichn[2]*self.expansion, ichn[2]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)
        self.emd_layer4=SpikConv(ichn[3]*self.expansion, ichn[3]*self.expansion, kernel_size=1,norm=norm,neuro=neuro)

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

    def init_rnn_output(self,B,H,W,T,device):
        outputs={'layer1':torch.zeros(B, self.ichn[0]*self.expansion,  H//4,  W//4, T, device=device), 
                 'layer2':torch.zeros(B, self.ichn[1]*self.expansion,  H//8,  W//8, T, device=device), 
                 'layer3':torch.zeros(B, self.ichn[2]*self.expansion, H//16, W//16, T, device=device), 
                 'layer4':torch.zeros(B, self.ichn[3]*self.expansion, H//32, W//32, T, device=device), 
                }
        return outputs
    
    def forward(self, xs, features=None, max_t=None):
        B,_,H,W,TS = xs.shape
        outputs_dict = self.init_rnn_output(B,H,W,TS+1,xs.device)
        self.h_and_c_previous = {k:None for k in ['layer1', 'layer2', 'layer3', 'layer4']}

        if features is not None:
            assert len(features)>=4, 'length of features must greater than 3'
            features = features[-4:]
            for i,name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
                emd_block = getattr(self,'emd_'+name)
                fea = features[i]
                fea = emd_block(fea)
                outputs_dict[name][...,0] = fea
        
        if max_t is None:
            max_t = TS 

        for t in range(max_t):
            x = xs[...,t]
            x = self.conv1(x)
            x = self.maxpool(x)

            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if not hasattr(self, name):
                    continue
                layer = getattr(self, name)
                x = layer(x)

                outputs_dict[name][...,t+1]=x

        out_layers = [outputs_dict[n] for n in self.out_features]
        return out_layers

# if __name__ == '__main__':
#     net = DarknetWrapper()
#     x=torch.rand((1,3,64,64))
#     y=net(x)
#     print(y)