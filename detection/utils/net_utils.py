import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional
import math
from collections import OrderedDict

def save_model(net, optim, scheduler, recorder, is_best=False, ckpt_name = None):
    model_dir = os.path.join(recorder.work_dir, 'ckpt')
    # os.system('mkdir -p {}'.format(model_dir))
    os.makedirs(model_dir, exist_ok=True)
    epoch = recorder.epoch
    if ckpt_name is None:
        ckpt_name = 'best' if is_best else epoch
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(ckpt_name)))

    # remove previous pretrained model if the number of models is too big
    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    # if len(pths) <= 2:
    #     return
    # os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network_specified(net, model_dir, logger=None, pth_save_prefix=None):
    if logger:
        logger.info('Load model specified from: ' + model_dir)
    pretrained_net = torch.load(model_dir)
    if 'net' in pretrained_net.keys():
        pretrained_net = pretrained_net['net']
    elif 'model' in pretrained_net.keys():
        pretrained_net = pretrained_net['model']
    elif 'state_dict' in pretrained_net.keys():
        pretrained_net = pretrained_net['state_dict']

    net_state = net.state_dict()
    state = {}
    if pth_save_prefix is None:
        for k, v in pretrained_net.items():
            if k not in net_state.keys() or v.size() != net_state[k].size():
                if logger:
                    logger.info('skip weights: ' + k)
                continue
            state[k] = v
    else:
        for k, v in pretrained_net.items():
            ak = pth_save_prefix + k
            if ak not in net_state.keys() or v.size() != net_state[ak].size():
                if logger:
                    logger.info('skip weights: ' + k)
                continue
            state[ak] = v  
        
    net.load_state_dict(state, strict=False)

    if logger:
        logger.info('Load model specified from: ' + model_dir)
    
    # net.state_dict()['module.backbone.model.conv1.weight']
    # pretrained_net['backbone.model.conv1.weight']

    # for k, v in pretrained_net.items():
    #     if k not in net_state.keys() or v.size() != net_state[k].size():
    #         if logger:
    #             logger.info('skip weights: ' + k)
    #         continue
    #     state[k] = v

    #     ak = 'module.'+ k
    #     if ak not in net_state.keys() or v.size() != net_state[ak].size():
    #         if logger:
    #             logger.info('skip weights: ' + k)
    #         continue
    #     state[ak] = v  


def load_network(net, model_dir, finetune_from=None, logger=None):
    if finetune_from:
        if logger:
            logger.info('Finetune model from: ' + finetune_from)
        load_network_specified(net, finetune_from, logger)
        return
    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'], strict=True)

def load_network_finetune(net, model_dir, logger=None):
    
    if model_dir.split('/')[-1] in ['yolov3tiny_voc.pth','best_epoch_weights.pth']:
        if logger:
            logger.info(f'--------------load model weights: {model_dir}')

        from pretrained.match_keys import yolov3tiny_voc_keys
        pretrained_net = torch.load(model_dir)
        net_state = net.state_dict()
        state = {}
        for k, v in pretrained_net.items():
            mk = yolov3tiny_voc_keys[k]
            if mk not in net_state.keys() or v.size() != net_state[mk].size():
                if logger:
                    logger.info('skip weights: ' + k)
                continue
            state[mk] = v
        net.load_state_dict(state, strict=False)
    else:
        if logger:
            logger.info(f'skip model weights: {model_dir}')

def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).\
                    reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).\
                    reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
