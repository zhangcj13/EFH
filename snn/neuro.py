import inspect
import six
from spikingjelly.clock_driven import surrogate
import torch
# from . import surrogate
from spikingjelly.clock_driven.neuron import BaseNode,IFNode,LIFNode
from typing import Callable

# THRESH = 0.5  # neuronal threshold
LENS = 0.5  # hyper-parameters of approximate function
DECAY = 0.5  # 0.2 # decay constants
class pseudo_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input, alpha)
        ctx.save_for_backward(input)
        return surrogate.heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < ctx.alpha
        temp = abs(input) < LENS
        # res= grad_input * temp.float()
        # print("temp: ",temp.float(),"restult: ",res)
        return grad_input * temp.float()


class PseudoSpike(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=0.5, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return pseudo_spike.apply(x)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        # mask0 = (x > (1.0 / alpha)).to(x)
        # mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        # return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)
        return x

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None
    
class MultiSpike(torch.nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x): # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)

class NonSpike(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x): # B C H W
        return x

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

surrogate_dict={
    'Sigmoid':surrogate.Sigmoid(),
    'Atan':surrogate.ATan(),
    'Erf':surrogate.Erf(),
    'PiecewiseLeakyReLU':surrogate.PiecewiseLeakyReLU(),
    'PiecewiseQuadratic':surrogate.PiecewiseQuadratic(),
    'SoftSign':surrogate.soft_sign(),
    'PiecewiseExp':surrogate.PiecewiseExp(),
    'PseudoSpike': PseudoSpike()
}

class ndLIFNode(LIFNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (self.v_reset - self.v) / self.tau + x
        
class NonSpikeIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        self.neuronal_fire()
        # self.neuronal_reset()
        return self.v




def get_neuro(param):
    assert isinstance(param, dict) and 'type' in param
    args = param.copy()
    obj_type = args.pop('type') 

    if 'surrogate_function' in args:
        if args['surrogate_function'] in surrogate_dict.keys():
            args['surrogate_function']=surrogate_dict[args['surrogate_function']]
        else:
            args['surrogate_function']=surrogate.Sigmoid()

    return  globals()[obj_type](**args)


class PIFNode(torch.nn.Module):
    def __init__(
        self,
        node
        ):
        super().__init__()
        assert node.min_value==0, f' min_value should be 0'
        self.v_threshold = 1.0
        self.v_reset = self.v_threshold
        self.v_scale =1.0

        self.T = int(node.max_value)

        self.v_init = 0.5
            
    def __repr__(self):
        return f"PIF (v_threshold={self.v_threshold}, v_reset={self.v_reset}, v_scale={self.v_scale})"     

    def forward(self, x): # B C H W
        v = x + self.v_init 
        for t in range(self.T):
            st=(v >= self.v_threshold).to(v)
            if t==0:
                s=st
            else:
                s+=st
            v=v-self.v_threshold

        # return s * self.v_scale
        return s

class NPIFNode(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.v_threshold = 1.0
        self.v_reset = 0
        self.v_scale =1.0
        self.T = 4
        self.v_init = 0

    def forward(self, x): # B C H W
        return x


def replace_ms_by_pif(model):
    for name, module in model._modules.items():
        # if isinstance(module,Conv):
        #     model._modules[name] = msBlock(module.conv,module.norm,module.act)
        #     continue
        if hasattr(module,"_modules"):
            model._modules[name] = replace_ms_by_pif(module)
        
        if module.__class__.__name__ == 'MultiSpike':
            model._modules[name] = PIFNode(model._modules[name])
        elif module.__class__.__name__ == 'NonSpike':
            model._modules[name] = NPIFNode()
    return model


