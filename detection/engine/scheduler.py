import torch
import math
from typing import List
from torch.optim.lr_scheduler import _LRScheduler, StepLR,CosineAnnealingWarmRestarts,CosineAnnealingLR,MultiStepLR
from functools import partial

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


def _yolox_warm_cos_lr(
    min_lr_ratio: float,
    total_iters: int,
    warmup_total_iters: int,
    warmup_lr_start: float,
    no_aug_iter: int,
    steps_at_iteration: List[int],
    reduction_at_step: float,
    iters: int)->float:
    """Cosine learning rate with warm up."""
    min_lr = min_lr_ratio
    if iters < warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (1 - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    else:
        lr = min_lr + 0.5 * (1 - min_lr) * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))

    for step in steps_at_iteration:
        if iters >= step:
            lr *= reduction_at_step
    return lr

# class XLRSchedule:
#     def __init__(self,
#                  warmup_epochs: float,
#                  num_iters_per_epoch: int,
#                  tot_num_epochs: int,
#                  min_lr_ratio: float=0.05,
#                  warmup_lr_start: float=0,
#                  steps_at_iteration=[50000],
#                  reduction_at_step=0.5):

#         warmup_total_iters = num_iters_per_epoch * warmup_epochs
#         total_iters = tot_num_epochs * num_iters_per_epoch
#         no_aug_iters = 0
#         self.lr_func = partial(_yolox_warm_cos_lr, min_lr_ratio, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iters, steps_at_iteration, reduction_at_step)

#     def __call__(self, *args, **kwargs)->float:
#         return self.lr_func(*args, **kwargs)

class YOLOXWarmCosLR(_LRScheduler):
    def __init__(self, optimizer, 
                 warmup_epochs: float,
                 num_iters_per_epoch: int,
                 tot_num_epochs: int,
                 min_lr_ratio: float=0.05,
                 warmup_lr_start: float=0,
                 steps_at_iteration=[50000],
                 reduction_at_step=0.5,
                 last_epoch=-1):
        warmup_total_iters = num_iters_per_epoch * warmup_epochs
        total_iters = tot_num_epochs * num_iters_per_epoch
        no_aug_iters = 0
        self.lr_func = partial(_yolox_warm_cos_lr, min_lr_ratio, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iters, steps_at_iteration, reduction_at_step)
        super(YOLOXWarmCosLR, self).__init__(optimizer, last_epoch)
 
    def get_lr(self):
        lr_ratio = self.lr_func(iters=self.last_epoch)
        return [base_lr*lr_ratio for base_lr in self.base_lrs]
        # return [base_lr*self.lr_func(iters=self.last_epoch) for base_lr in self.base_lrs]

def build_scheduler(cfg, optimizer):

    cfg_cp = cfg.scheduler.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type=='PolyLR':
        return PolyLR(optimizer, **cfg_cp)
    elif cfg_type=='YOLOXWarmCosLR':
        return YOLOXWarmCosLR(optimizer, **cfg_cp)

    if cfg_type not in dir(torch.optim.lr_scheduler):
        raise ValueError("{} is not defined.".format(cfg_type))

    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type) 

    return _scheduler(optimizer, **cfg_cp) 

if __name__ == "__main__":
     
    import torch.optim as optim
    import torch
    import matplotlib.pyplot as plt
    net=torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,),
        torch.nn.ReLU()
    )
    epochs = 300 #100
    num_iters_per_epoch = 16551 // 64
    total_iter = num_iters_per_epoch * epochs

    optimizer = optim.SGD(net.parameters(), 0.01, momentum = 0.99, nesterov=True)

    scheduler=YOLOXWarmCosLR(
            optimizer, 
            warmup_epochs=5,
            num_iters_per_epoch=num_iters_per_epoch,
            tot_num_epochs=epochs,
            min_lr_ratio=0.05,
            warmup_lr_start=0,
            steps_at_iteration=[int(total_iter*0.9)],
            reduction_at_step=0.5,
        )

    optimizer1 = optim.SGD(net.parameters(), 0.01, momentum = 0.98, nesterov=True)    
    scheduler1=StepLR(
            optimizer1, 
            step_size=1, gamma=0.991
        )
    
    optimizer2 = optim.SGD(net.parameters(), 0.01, momentum = 0.98, nesterov=True)    
    scheduler2=MultiStepLR(
            optimizer2, 
            milestones=[60, 120, 200], gamma=0.5
        )
    
    # scheduler = CosineAnnealingLR(optimizer,T_max=5*16, eta_min=0, last_epoch=-1, verbose=False)

    def get_lr_scheduler(optimer, scheduler, total_step, num_iters_per_epoch=1,scale=1.0):
        '''
        get lr values
        '''
        lrs = []
        for step in range(total_step):
            lr_current = optimer.param_groups[0]['lr']
            lrs.append(lr_current*scale)
            if scheduler is not None:
                if step%num_iters_per_epoch==0:
                    scheduler.step()
        return lrs
    
    plt.clf()
    lrs = get_lr_scheduler(optimizer, scheduler, total_iter)
    lrs1 = get_lr_scheduler(optimizer1, scheduler1, total_iter,num_iters_per_epoch)
    lrs2 = get_lr_scheduler(optimizer2, scheduler2, total_iter,num_iters_per_epoch)
    plt.plot(lrs, label='YOLOXWarmCosLR')
    plt.plot(lrs1, label='StepLR')
    plt.plot(lrs2, label='MultiStepLR')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')


    plt.title('LRScheduler')
    plt.legend()
    # plt.show()
    plt.savefig('output_figure.png')
