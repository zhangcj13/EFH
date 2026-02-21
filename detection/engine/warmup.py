import pytorch_warmup as warmup

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

class SquareWarmup(warmup.BaseWarmup):

    def __init__(self, optimizer, warmup_period, last_step=-1,scale=0.1):
        group_count = len(optimizer.param_groups)
        warmup_params = warmup.get_warmup_params(warmup_period, group_count)
        self.scale = min(1.0,max(0, scale))
        super(SquareWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return min(1.0,(1 - self.scale) * pow((step+1) / float(warmup_period), 2)+ self.scale)


warmup_dict={
    'SquareWarmup':SquareWarmup,
    'ExponentialWarmup':warmup.ExponentialWarmup,
    'LinearWarmup':warmup.LinearWarmup
}

def build_warmup(cfg, optimizer):
    cfg_cp = cfg.warmup.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in warmup_dict.keys():
        raise ValueError("warmup {} is not defined.".format(cfg_type))

    return warmup_dict[cfg_type](optimizer, **cfg_cp) 