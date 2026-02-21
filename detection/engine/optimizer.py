import torch
import torch.nn as nn


def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(torch.optim, cfg_type)
    if hasattr(cfg,'paramgroup'):
        pg_type=cfg.paramgroup
        if pg_type=='weight_bias_bn':
            
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            for k, v in net.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = _optim(pg0, **cfg_cp)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay":  5e-4}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            return  optimizer
    
    return _optim(net.parameters(), **cfg_cp)

    
