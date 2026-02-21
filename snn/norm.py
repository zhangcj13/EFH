
from torch.nn import BatchNorm2d
from .stbp.helper_layers import tdBatchNorm

norm_dict={
    'BN':BatchNorm2d,
    'tdBN':tdBatchNorm,
}

# def get_norm(param):
#     assert isinstance(param, dict) and 'type' in param
#     args = param.copy()
#     obj_type = args.pop('type') 
#     if obj_type is None or obj_type not in norm_dict.keys():
#         return None
#     return norm_dict[obj_type](**args)

def get_norm(param, num_features):
    assert isinstance(param, dict) and 'type' in param
    args = param.copy()
    obj_type = args.pop('type') 
    if obj_type is None or obj_type not in norm_dict.keys():
        return None
    args['num_features']=num_features
    return norm_dict[obj_type](**args)