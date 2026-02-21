from detection.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')
AGGREGATORS = Registry('aggregators')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')
LOSSES = Registry('loss')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))

def build_separate_backbones(backbone_cfg, cfg=None):
    return build(backbone_cfg, BACKBONES, default_args=dict(cfg=cfg) if cfg is not None else None)

def build_aggregator(cfg):
    return build(cfg.aggregator, AGGREGATORS, default_args=dict(cfg=cfg))

def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))

def build_head(split_cfg, cfg=None):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg) if cfg is not None else None)

def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))

def build_necks(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))

def build_separate_necks(neck_cfg,cfg=None):
    return build(neck_cfg, NECKS, default_args=dict(cfg=cfg) if cfg is not None else None)

def build_loss(cfg):
    """Build loss."""
    # return build(cfg, LOSSES, default_args=dict(cfg=cfg))
    return build(cfg, LOSSES)
