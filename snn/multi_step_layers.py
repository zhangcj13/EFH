import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange

from .qcfs.modules import ScaledNeuron
from spikingjelly.clock_driven import neuron

def replace_ss_by_ms(model, neuron='tdIF',delay=None):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_ss_by_ms(module, neuron=neuron,delay=delay)
        
        if module.__class__.__name__ == 'Conv2d':            
            layer = msConv(in_channels = module.in_channels,
                        out_channels= module.out_channels,
                        kernel_size= module.kernel_size,
                        stride= module.stride,
                        padding= module.padding,
                        dilation= module.dilation,
                        groups= module.groups,
                        bias= module.bias is not None,)
            
            if layer.bias is None and module.bias is not None:
                layer.bias = nn.Parameter(module.bias)
            else:
                layer.bias = module.bias
            layer.weight.data = module.weight.data

            model._modules[name] = layer
        
        elif module.__class__.__name__ == 'BatchNorm2d':
            layer = msBatchNorm(num_features = module.num_features,)

            layer.weight.data.copy_(module.weight.data)
            layer.bias.data.copy_(module.bias.data)
            layer.running_mean.data.copy_(module.running_mean.data)
            layer.running_var.data.copy_(module.running_var.data)
            layer.num_batches_tracked.data.copy_(module.num_batches_tracked.data)

            model._modules[name]=layer
        
        elif module.__class__.__name__ == 'AvgPool2d':
            model._modules[name] = msAvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
        elif module.__class__.__name__ == 'AdaptiveAvgPool2d':
            model._modules[name] = msAdaptiveAvgPool2d(output_size=module.output_size)
        elif module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = msMaxPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
        elif module.__class__.__name__ == 'ZeroPad2d':
            model._modules[name] = msZeroPad2d(module.padding)
        elif module.__class__.__name__ == 'Upsample':
            model._modules[name] = msUpsample(scale_factor=module.scale_factor, mode=module.mode)

        elif module.__class__.__name__ == 'ScaledNeuron':
           assert neuron in ['IF','A2F','tdIF'],  'neuron type not in (IF, A2F,tdIF)'
           if neuron=='IF':
               model._modules[name] = msScaledNeuron(scale=module.scale)
           elif neuron=='A2F':
               model._modules[name] = msScaledNeuronA2F(scale=module.scale,delay=delay)
           elif neuron=='tdIF':
               model._modules[name] = tdScaledNeuron(scale=module.scale,delay=delay)
        #    model._modules[name] = msScaledNeuron(scale=module.scale)
        #    model._modules[name] = msScaledNeuronA2F(scale=module.scale)
        #    model._modules[name] = tdScaledNeuron(scale=module.scale)
    return model

class msBlock(nn.Module):
    """Standard Convolutional Block"""
    def __init__(self, block, norm = None, act = None, o='FS',time_dependent=False) -> None:
        super().__init__()

        self.block =  block
        self.norm = norm
        self.act = act
        self.o=o
        self.time_dependent = time_dependent
    
    def time_forward(self,x):
        steps = x.shape[-1]
        for step in range(steps):
            y = self.block(x[..., step])
            if self.norm:
                y = self.norm(y)
            if self.act:
                y = self.act(y)
            if step==0:
                x_ = torch.zeros(y.shape + (steps,), device=x.device)# Add time dimension to the last dim
            x_[..., step] = y
        return x_

    def seq_forward(self,x):
        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
                
        y = self.block(y)
        if self.norm:
            y = self.norm(y)
        if self.act:
            y = self.act(y)
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"input does  not have time step!"
        if self.time_dependent:
            x_=self.time_forward(x)
        else:
            x_=self.seq_forward(x)

        if self.o=='mean':
            return torch.mean(x_, dim=-1, keepdim=False)
        elif self.o=='sum':
            return torch.sum(x_, dim=-1, keepdim=False)
        elif self.o=='last':
            return x_[..., -1]
        else:
            return x_

class msConv(nn.Conv2d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
                
        y = self._conv_forward(y, self.weight, self.bias)
    
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_
        # return self._conv_forward(x, self.weight, self.bias)

class msBatchNorm(nn.BatchNorm2d):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)               

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        y = F.batch_norm(
                y,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
    
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_

class msAvgPool2d(nn.AvgPool2d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)

        y = F.avg_pool2d(y, self.kernel_size, self.stride,
                        self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_

class msAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)

        y = F.adaptive_avg_pool2d(y, self.output_size)
          
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_

class msMaxPool2d(nn.MaxPool2d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
                
        y = F.max_pool2d(y, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
    
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_
    
class msZeroPad2d(nn.ZeroPad2d):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)

        y = F.pad(y, self.padding, 'constant', self.value)
 
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_
    
class msUpsample(nn.Upsample):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        self._check_input_dim(x)
        x_seq=x.permute(4,0,1,2,3)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)

        y =  F.interpolate(y, self.size, self.scale_factor, self.mode, self.align_corners)
 
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)
        x_= y.permute(1,2,3,4,0)
        return x_

class msScaledNeuron(ScaledNeuron):
    def forward(self, xs):  
        assert xs.dim() == 5, f"input does  not have time step!"
        ts = xs.shape[-1]

        for t in range(ts):
            x = xs[...,t]

            y = x / self.scale
            if t == 0:
                self.neuron(torch.ones_like(y)*0.5)
            y = self.neuron(y)
            y = y * self.scale

            if t==0:
                x_ = torch.zeros(y.shape + (ts,), device=xs.device)# Add time dimension to the last dim
            x_[..., t] = y
        return x_

class msScaledNeuronA2F(ScaledNeuron):    
    def __init__(self, scale=1.,delay=None):
        super(msScaledNeuronA2F, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
        self.ms = False 
        self.delay=delay

    def forward(self, xs):  
        assert xs.dim() == 5, f"input does  not have time step!"

        ts = xs.shape[-1]
        if self.delay is None:
            self.delay = ts

        dt=0
        for t in range(ts):
            x=xs[...,t]
            x = x / self.scale
            if t == 0:
                self.neuron(torch.ones_like(x)*0.5)
            self.neuron.neuronal_charge(x)

            if t>=self.delay:
                self.neuron.neuronal_fire()
                self.neuron.neuronal_reset()
                y = self.neuron.spike #* self.scale
                if dt==0:
                    x_ = torch.zeros(y.shape + (ts,), device=xs.device)# Add time dimension to the last dim
                x_[..., dt] = y
                dt+=1
        
        for t in range(dt, ts):
            self.neuron.neuronal_fire()
            self.neuron.neuronal_reset()
            y = self.neuron.spike #* self.scale
            if t==0:
                x_ = torch.zeros(y.shape + (ts,), device=xs.device)# Add time dimension to the last dim
            x_[..., t] = y
        
        self.cal_spike_rate(x_)
        x_ = x_ * self.scale

        return x_

    def cal_spike_rate(self,ys):
        if not hasattr(self,'spike_counts'):
            self.spike_counts=torch.zeros_like(ys)
            self.cal_time=0
        
        if  self.cal_time>32:
            return

        self.spike_counts += ys
        self.cal_time +=1

        if  self.cal_time==32:
            n,c,h,w,t = self.spike_counts.shape
            total_neuros=c*h*w*t 
            all_spikes = torch.sum(self.spike_counts)/self.cal_time
            spike_rate=all_spikes/total_neuros
            # print(f'total_neuros: {total_neuros/t}, spike_rate: {spike_rate}')
            print(f'[{c},{h},{w},{total_neuros/t}, {spike_rate}],')
    
        # ''' save spike_rate for analysis '''
        # spike=x_/self.scale
        # shape=spike.shape
        # total_spike=torch.sum(spike)
        # total_neurons=shape[0]*shape[1]*shape[2]*shape[3]*shape[4]
        # spike_rate=total_spike/total_neurons
        # # print(f'spike_rate:{spike_rate}')

        # if not hasattr(self,'spike_rate'):
        #     self.spike_rate=0
        # self.spike_rate=spike_rate
        # self.total_neurons=total_neurons

        # return x_
    
class tdIFNode(neuron.IFNode):
    def neuronal_charge(self, x: torch.Tensor,r=1.0):
        self.v = self.v + x*r

    def neuronal_fire(self,r=1.0):
        self.spike = self.surrogate_function(self.v - self.v_threshold*r)

    def neuronal_reset(self,r=1.0):

        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold*r
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    # def forward(self, xs: torch.Tensor):
    #     assert xs.dim() == 5, f"input does  not have time step!"
    #     ts = xs.shape[-1]
        
    #     for t in range(ts):
    #         x = xs[...,t]
    #         self.neuronal_charge(x)
        
    #     for t in range(ts):
    #         self.neuronal_fire()
    #         self.neuronal_reset()
    #         if t==0:
    #             x_ = torch.zeros(self.spike.shape + (ts,), device=self.spike.device)# Add time dimension to the last dim
    #         x_[..., t] = self.spike

    #     return x_
    
    def forward(self, xs: torch.Tensor,delay=0):
        assert xs.dim() == 5, f"input does  not have time step!"
        ts = xs.shape[-1]
        rs = [2**(ts-1-t) for t in range(ts)]

        dt=0
        for t in range(ts):
            x = xs[...,t]
            r = rs[t]
            self.neuronal_charge(x,r)

            if t>=delay:
                rfire = rs[dt]
                self.neuronal_fire(rfire)
                self.neuronal_reset(rfire)
                if dt==0:
                    x_ = torch.zeros(self.spike.shape + (ts,), device=xs.device)# Add time dimension to the last dim
                x_[..., dt] = self.spike
                dt+=1
        
        for t in range(dt,ts):
            r = rs[t]
            self.neuronal_fire(r)
            self.neuronal_reset(r)
            if t==0:
                x_ = torch.zeros(self.spike.shape + (ts,), device=self.spike.device)# Add time dimension to the last dim
            x_[..., t] = self.spike

        return x_

class tdScaledNeuron(ScaledNeuron):
    def __init__(self, scale=1.,delay=None):
        super(tdScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = tdIFNode(v_reset=None)
        self.delay=delay
    def forward(self, xs):  
        assert xs.dim() == 5, f"input does  not have time step!"

        ts = xs.shape[-1]
        if self.delay is None:
            self.delay = ts

        ys = xs / self.scale
        self.neuron.neuronal_charge(torch.ones_like(ys[...,0])*0.5)
        ys = self.neuron(ys, delay=self.delay)

        # self.cal_spike_rate(ys)

        ys = ys * self.scale
        return ys
    
    def cal_spike_rate(self,ys):
        if not hasattr(self,'spike_counts'):
            self.spike_counts=torch.zeros_like(ys)
            self.cal_time=0
        
        if  self.cal_time>32:
            return
        
        self.spike_counts += ys
        self.cal_time +=1

        if  self.cal_time==32:
            n,c,h,w,t = self.spike_counts.shape
            total_neuros=c*h*w*t 
            all_spikes = torch.sum(self.spike_counts)/self.cal_time
            spike_rate=all_spikes/total_neuros
            # print(f'total_neuros: {total_neuros/t}, spike_rate: {spike_rate}')
            print(f'[{c},{h},{w},{total_neuros/t}, {spike_rate}],')



def get_spike_rate_from_neuron(model,parent_name='',spike_rate={}):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = get_spike_rate_from_neuron(module,parent_name=parent_name+'.'+name,spike_rate=spike_rate)
        if module.__class__.__name__ == 'msScaledNeuronA2F':
        #    spike_rate[parent_name+'.'+name]=module.spike_rate.cpu().numpy()
           spike_rate[parent_name+'.'+name]=module.total_neurons
        #    model._modules[name] = msScaledNeuronA2F(scale=module.scale)
    return model

