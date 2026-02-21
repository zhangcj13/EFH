import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_scale2conv(conv, scale):
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    fusedconv.weight.copy_((w_conv*scale).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    fusedconv.bias.copy_(b_conv*scale)

    return fusedconv

def fuse_conv_and_bn_list(nnlist):
    for i,m in enumerate(nnlist):
        if i >= len(nnlist)-1:
            break
        if isinstance(m, (nn.Conv2d,nn.Linear)) and isinstance(m, (nn.BatchNorm2d)):
        # if type(m)==nn.Conv2d and type(nnlist[i+1])==nn.BatchNorm2d:
            nnlist[i] = fuse_conv_and_bn(nnlist[i], nnlist[i+1])
            nnlist[i+1] = nn.Identity()

def fuse_net(model):
    for name, module in model._modules.items():
        if hasattr(module, 'fuse'):
            module.fuse()
        if hasattr(module,"_modules"):
            fuse_net(module)
        # if 'fuse' in module.__class__.__name__:
        
    return model