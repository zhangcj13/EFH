# encoding: utf-8
import torch
  
def eval_on_dsec(exp,net,maxT=10,split='val',batch_size=16):
    evaluator = exp.get_evaluator( batch_size=batch_size, is_distributed=False,split=split, ts=maxT)
    outputs = evaluator.evaluate_all_step(net, False, False, return_outputs=True, total_times = maxT+1)
    return outputs


if __name__ == '__main__':
    from detection.utils.config import Config
    from configs.yolox_exp.RsHN_DSEC import RsHN_Exp

    ''' build exp  '''
    cfg = Config.fromfile('configs/DsecDet/cfg_RsHN_ms_eval.py')
    texp=RsHN_Exp(cfg)

    ''' get model and load weight  '''
    net = texp.get_model()

    weight_path = './YOLOX_outputs/RsHN_DSEC2/RsHNet/resnet18t/TS10_RSR_CDFNFPN_hwMosaic_kk975/best_ckpt.pth'

    state_dict = torch.load(weight_path)
    print(f'epoch: {state_dict["start_epoch"]}, best_mAP: {state_dict["best_ap"]}')
    net.load_state_dict(state_dict['model'], strict=True)
    net.cuda()
    net.eval()

    print('--------- start eval ---------')
    result=eval_on_dsec(texp, net, maxT=cfg.ts, split='test',batch_size=128)
    print(result)
