# encoding: utf-8
import sys
import time
import os
from yolox.data import get_yolox_datadir
from yolox.exp import Exp as BaseExp
from yolox.data import TrainTransform_FaE, ValTransform_FaE
from yolox.data import DsecIFD

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from pathlib import Path

from detection.models.registry import build_backbones, build_aggregator, build_heads, build_necks,build_head,build_separate_backbones
from spikingjelly.clock_driven import functional
from einops import rearrange, repeat

class ResYOLOX(nn.Module):
    def __init__(self, cfg):
        super(ResYOLOX, self).__init__()
        
        self.cfg = cfg

        self.backbone = build_separate_backbones(cfg.frame_backbone, cfg)

        self.neck = build_necks(cfg)
        
        cfg['forawd_yolox']=True

        self.bbox_heads = build_head(cfg.bbox_heads, cfg)
    
    def forward(self, data, targets=None):
        
        x = data['image']
        frame_features = self.backbone(x)
        fpn_outs = self.neck(frame_features)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.bbox_heads(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.bbox_heads(fpn_outs)
        return outputs

    def visualize(self, data, targets, save_prefix="assign_vis_"):
        x = data['image']
    

class RsHNet(nn.Module):
    def __init__(self, cfg):
        super(RsHNet, self).__init__()
        
        self.cfg = cfg

        self.backbone = build_separate_backbones(cfg.frame_backbone, cfg)
        self.event_backbone = build_separate_backbones(cfg.event_backbone, cfg)

        self.neck = build_necks(cfg)
        
        cfg['forawd_yolox']=True

        self.bbox_heads = build_head(cfg.bbox_heads, cfg)
    
    def forward(self, data, targets=None):
        functional.reset_net(self.event_backbone)
        # fpn output content features of [dark3, dark4, dark5]
        x = data['image']
        evt = data['event']
        seq_length = data['seq_length']

        # x = data[0]
        # evt = data[1]
        # seq_length = data[2]
        # x = data

        frame_features = self.backbone(x)

        # return frame_features[-1]
        
        max_t = torch.max(seq_length)
        event_features = self.event_backbone(evt,frame_features,max_t)

        fpn_outs = self.neck(frame_features, event_features, seq_length)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.bbox_heads(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.bbox_heads(fpn_outs)
        # outputs = self.bbox_heads(fpn_outs)
        return outputs

    def visualize(self, data, targets, save_prefix="assign_vis_"):
        x = data['image']
        evt = data['events']

        # fpn_outs = self.backbone(x)
        # self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
    
    def infer_all_steps(self, data, step=10):
        functional.reset_net(self.event_backbone)
        # fpn output content features of [dark3, dark4, dark5]
        x = data['image']
        evt = data['event']
        seq_length = data['seq_length']

        frame_features = self.backbone(x)

        max_t = torch.max(seq_length)
        event_features = self.event_backbone(evt,frame_features,max_t)
        
        outputs=[]
        for n in range(step):
            tmp_seq_length=torch.from_numpy(np.array([n])).to(x.device)
            # tmp_seq_length=torch.from_numpy(np.array([0])).to(x.device)
            fpn_outs = self.neck(frame_features, event_features, tmp_seq_length)
            outputs.append(self.bbox_heads(fpn_outs))
        
        # outputs=torch.cat(outputs,dim=0)
        return outputs
    
class RsHN_Exp(BaseExp):
    def __init__(self, cfg):
        super(RsHN_Exp, self).__init__()
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.50

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.input_size = cfg.input_size if hasattr(cfg,'input_size') else (224, 320)  # (height, width)
        self.test_size =  cfg.test_size if hasattr(cfg,'test_size') else (224, 320)

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = cfg.warmup_epochs if hasattr(cfg,'warmup_epochs') else 10
        # max training epoch
        self.max_epoch = cfg.max_epoch if hasattr(cfg,'max_epoch') else 90  # ccccccccccccccccccccccccccccccccccc> 200 90
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = cfg.lr_all/64.0 if hasattr(cfg,'lr_all') else  0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = cfg.no_aug_epochs if hasattr(cfg,'no_aug_epochs') else 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = cfg.print_interval if hasattr(cfg,'print_interval') else 50
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = cfg.eval_interval if hasattr(cfg,'eval_interval') else 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # train with out events
        self.no_event_epoch = cfg.no_event_epoch if hasattr(cfg,'no_event_epoch') else 0
        # freeze image backbone when start to train with event
        self.freeze_image_backbone_epochs = cfg.freeze_image_backbone_epochs if hasattr(cfg,'freeze_image_backbone_epochs') else [-10,-1]

        self.multiscale_range = cfg.multiscale_range if hasattr(cfg,'multiscale_range') else 5
        self.evt_dt= cfg.evt_dt if hasattr(cfg,'evt_dt') else 5000
        self.eval_us = cfg.eval_us if hasattr(cfg,'eval_us') else 50000
        self.enable_mosaic = cfg.enable_mosaic if hasattr(cfg,'enable_mosaic') else False

        self.cfg = cfg

        self.output_dir = "./YOLOX_outputs/RsHN_DSEC2/"+ self.cfg.net_type + '/' + cfg.arch_net

        # self.data_num_workers=1

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if self.cfg.net_type == 'RsHNet':
            self.model = RsHNet(self.cfg)
        elif self.cfg.net_type == 'ResYOLOX':
            self.model = ResYOLOX(self.cfg)
            
        self.model.apply(init_yolo)
        self.model.bbox_heads.initialize_biases(1e-2)
        
        #*************************
        # load pre trained model
        #*************************
        pretrain_pth = self.cfg.pre_weight if hasattr(self.cfg,'pre_weight') else None
        if pretrain_pth is not None:
            ckpt_info = torch.load(pretrain_pth)
            print('********** start epoch:', ckpt_info['start_epoch'])

            pretrained_net = ckpt_info['model']
            net_state = self.model.state_dict()
            state = {}
            for k, v in pretrained_net.items():
                if k not in net_state.keys() or v.size() != net_state[k].size():
                    print('skip weights: ' + k)
                    continue
                state[k] = v
            self.model.load_state_dict(state, strict=False)
        
        #*************************
        # load pre trained sew weights
        #*************************
        if hasattr(self.cfg,'pre_sew') and self.cfg.pre_sew:
            print('>>>>>>>>>>>>>>>>>>> load pre trained sew weight <<<<<<<<<<<<<<<<<<<')
            self.model.event_backbone.load_pretrained()

        self.model.train()
        return self.model

    def preprocess(self, images, events, targets, seq_length,tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if events.shape[1]==0:
            events = None
        if scale_x != 1 or scale_y != 1:
            images = nn.functional.interpolate(
                images, size=tsize, mode="bilinear", align_corners=False
            )
            if events is not None:
                _,C,_,_,T=events.shape
                events = rearrange(events, 'b c h w t->b (t c) h w')
                events = torch.nn.functional.interpolate(events, size=tsize, mode="nearest")
                events = rearrange(events, 'b (t c) h w->b c h w t',t=T, c=C)

            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        
        inputs ={'image':images,
                 'event':events,
                 'seq_length':seq_length}

        return inputs, targets

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        
        if  self.cfg.train_scale==2:
            # train_form =TrainTransform_FaE(max_labels=self.cfg.max_labels, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob,
            #                                scale=(0.5, 2),
            #                                min_bbox_diag=self.cfg.min_bbox_diag*0.6,
            #                                min_bbox_height=self.cfg.min_bbox_height*0.6,
            #                                noise_prob=0.0)
            train_form =TrainTransform_FaE(max_labels=self.cfg.max_labels, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob,
                                        #    scale=(0.5, 1.5),
                                           scale=(0.8, 1.2),
                                           degrees=0.0,
                                           translate=0.1,shear=0.0, #2.0
                                           min_bbox_diag=self.cfg.min_bbox_diag*0.6,
                                           min_bbox_height=self.cfg.min_bbox_height*0.6,
                                           noise_prob=0.0)
        else:
            train_form =TrainTransform_FaE(max_labels=self.cfg.max_labels, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob,
                                           scale=(0.5, 1.5),
                                           min_bbox_diag=self.cfg.min_bbox_diag*0.6,
                                           min_bbox_height=self.cfg.min_bbox_height*0.6,
                                           noise_prob=0.0)
        # train_form=ValTransform_FaE(legacy=False,return_targets=True)

        train_dataset = DsecIFD(root=Path(self.cfg.data_root), 
                       split='train',#"train", full
                       transform = train_form,
                       debug=False, 
                       min_bbox_diag   = self.cfg.min_bbox_diag/ self.cfg.train_min_bbox_scale, 
                       min_bbox_height = self.cfg.min_bbox_height/ self.cfg.train_min_bbox_scale,
                       scale=self.cfg.train_scale,
                       cropped_height=self.cfg.cropped_height,
                       num_us=self.evt_dt*self.cfg.total_time_steps,
                       dt=self.evt_dt,
                       random_us=True,
                       enable_mosaic=self.enable_mosaic,
                       )
        return train_dataset
      
    def get_data_loader(self, batch_size, is_distributed, no_aug = False, cache_img: str = None):
        from yolox.utils import wait_for_the_master
        from yolox.data import (
            YoloBatchSampler,
            # DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
    
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        # dataloader_kwargs = {"num_workers": 0, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
  
    def get_eval_dataset(self, **kwargs):    
        from yolox.data.datasets.westlake_campus import WLC_Data    
        legacy = kwargs.get("legacy", False)
        split  = kwargs.get("split", "val")
        fix_ts = kwargs.get("fix_ts", None)
        if fix_ts is None:
            eval_set= DsecIFD(root=Path(self.cfg.data_root), 
                        split=split,
                        transform = ValTransform_FaE(legacy=legacy),
                        debug=False, 
                        min_bbox_diag=self.cfg.min_bbox_diag, 
                        min_bbox_height=self.cfg.min_bbox_height,
                        scale=self.cfg.val_scale,
                        cropped_height=self.cfg.cropped_height,
                        num_us=self.evt_dt*self.cfg.total_time_steps,
                        dt=self.evt_dt,
                        # only_perfect_tracks=True,
                        random_us=False,)
        else:
            eval_set = DsecIFD(root=Path(self.cfg.data_root), 
                        split=split,
                        transform = ValTransform_FaE(legacy=legacy),
                        debug=False, 
                        min_bbox_diag=self.cfg.min_bbox_diag, 
                        min_bbox_height=self.cfg.min_bbox_height,
                        scale=self.cfg.val_scale,
                        cropped_height=self.cfg.cropped_height,
                        num_us=self.evt_dt*self.cfg.total_time_steps,
                        dt=self.evt_dt,
                        # only_perfect_tracks=True,
                        random_us=False,)
            
            eval_set.reset_us(us=fix_ts*self.evt_dt,random_us=False)
                
        return eval_set

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
    
    def get_test_dataset(self, **kwargs):       
        legacy = kwargs.get("legacy", False)

        return DsecIFD(root=Path(self.cfg.data_root), 
                       split="test",
                       transform = ValTransform_FaE(legacy=legacy),
                       debug=False, 
                       min_bbox_diag=self.cfg.min_bbox_diag, 
                       min_bbox_height=self.cfg.min_bbox_height,
                       scale=self.cfg.val_scale,
                       cropped_height=self.cfg.cropped_height,
                       num_us=self.evt_dt*self.cfg.total_time_steps,
                       dt=self.evt_dt,
                       random_us=False,)

    
    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, is_distributed, half, return_outputs=return_outputs)
    
    def get_trainer(self, args):
        # from yolox.core import Trainer
        from yolox.core import Trainer_frm_evt_RT as Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False, split='val', ts=None):
        from yolox.evaluators import FaSeqE_DSECEvaluator as DSECEvaluator
        # from yolox.evaluators import DSECEvaluator

        return DSECEvaluator(
                # dataloader=self.get_eval_loader(batch_size, is_distributed,
                #                                 testdev=testdev, legacy=legacy),
                dataloader=self.get_eval_loader(batch_size, is_distributed,
                                                testdev=testdev, legacy=legacy, split=split, fix_ts=ts),
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                # need_postprocess=False,
            )
