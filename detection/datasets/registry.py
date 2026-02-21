from detection.utils import Registry, build_from_cfg

import torch
import torch.nn as nn
from functools import partial
import numpy as np
import random
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from torch.utils.data import DistributedSampler

import copy

DATASETS = Registry('datasets')
PROCESS = Registry('process')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    # elif cfg['type'] == 'RepeatDataset':
    #     return RepeatDataset(build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataset(split_cfg, cfg=None):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg) if cfg is not None else None)

def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(split_cfg, cfg, is_train=True,drop_last=False):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)

    # dataset.__getitem__(1)

    init_fn = partial(
            worker_init_fn, seed=cfg.seed)

    samples_per_gpu = cfg.batch_size // cfg.gpus

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = cfg.batch_size, shuffle = shuffle,
        num_workers = cfg.workers, pin_memory = False, drop_last = drop_last,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn)

    return data_loader

def worker_init_fn_m(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader_m(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn_m, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader



