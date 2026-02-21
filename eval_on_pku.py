import torch
from tqdm import tqdm

def eval_on_pku(exp,net,maxT=10,split='val',batch_size=16):

    outputs={}
    for tts in tqdm([0, maxT]):
        evaluator = exp.get_evaluator( batch_size=batch_size, is_distributed=False,split=split, ts=tts)
        res = evaluator.evaluate(net, False, False, return_outputs=True)
        outputs[tts]={'metric': res[0][3]}

    return outputs


if __name__ == '__main__':
    from detection.utils.config import Config
    from configs.yolox_exp.RsHN_PKUSOD import RsHN_Exp


    ''' build exp  '''
    cfg = Config.fromfile('./configs/PKUSOD/cfg_RsHN.py')

    texp=RsHN_Exp(cfg)

    ''' get model and load weight  '''
    net = texp.get_model()


    weight_path = './YOLOX_outputs/RsHN_PKUSOD/RsHNet/resnet34t/TS8_RSR_CDFNFPN_hwMosaic_kk975/best_ckpt.pth'
    print(weight_path)

    state = torch.load(weight_path)['model']
    net.load_state_dict(state, strict=False)
    net.cuda()
    net.eval()
    # print(net)  

    
    print('--------- start eval ---------')
    result=eval_on_pku(texp, net, maxT=cfg.ts, split='test',batch_size=32)
    print(result)


    