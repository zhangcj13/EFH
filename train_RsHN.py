#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import importlib
import sys
import os

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices
from detection.utils.config import Config


def make_args():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # parser.add_argument('--config', default = 'configs/PKUSOD/cfg_RsHN.py', help='train config file path')
    parser.add_argument('--config', default = 'configs/DsecDet/cfg_RsHN_ms.py', help='train config file path')

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")#64
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/root/data1/ws/SNN_CV/configs/yolox_exp/yolo_exp.py',# None
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"  # False  True
    )
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False, 
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,  # False  True
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args

def get_exp_with_cfg(exp_file, cfg):
    sys.path.append(os.path.dirname(exp_file))
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    exp = current_exp.RsHN_Exp(cfg)
    return exp


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()

if __name__ == "__main__":
    configure_module()
    args = make_args()

    cfg = Config.fromfile(args.config)

    args.exp_file=cfg.exp_file
    args.batch_size=cfg.batch_size
    args.ckpt=cfg.ckpt

    exp = get_exp_with_cfg(args.exp_file, cfg)
    exp.merge(args.opts)

    if not args.experiment_name:        
        args.experiment_name = f'TS{cfg.total_time_steps}'

        if cfg.event_backbone.type=='SewResnetSDRNN2':
            args.experiment_name += '_RSR'
        elif cfg.event_backbone.type=='SewResnetNRNN':
            args.experiment_name += '_SEW''
        else:
            args.experiment_name = f'_{cfg.event_backbone.type}'
        
        if cfg.neck.type=='ReduceDDFPN':
            args.experiment_name += f'_RDU{cfg.ReduceChn[0]}x{cfg.ReduceChn[1]}x{cfg.ReduceChn[2]}'
        elif cfg.neck.type=='ReduceDDPM':
            args.experiment_name += f'_RDD{cfg.ReduceChn[0]}x{cfg.ReduceChn[1]}x{cfg.ReduceChn[2]}'
        else:
            args.experiment_name += f'_{cfg.neck.type}'
        
        args.experiment_name += '_hwMosaic_kk975'


        

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
