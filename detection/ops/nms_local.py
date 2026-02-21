import numpy as np
import torch

# Hard-coded maximum. Increase if needed.
MAX_COL_BLOCKS = 1000
STRIDE = 4
N_OFFSETS = 72  # if you use more than 73 offsets you will have to adjust this value
N_STRIPS = (N_OFFSETS - 1)
PROP_SIZE = (5 + N_OFFSETS)
DATASET_OFFSET = 0


def CHECK_INPUT(x):
    return x.type().is_cuda() and x.is_contiguous()

def DIVUP(m, n):
    return (((m) + (n) - 1) / (n))

def bool_to_float(bool_value):
    return 1.0 if bool_value else 0.0

def devIoU(a, b, threshold: float = 0.5):
    start_a = int(a[2] * N_STRIPS - DATASET_OFFSET + 0.5)  # 0.5 rounding trick
    start_b = int(b[2] * N_STRIPS - DATASET_OFFSET + 0.5)
    start = max(start_a, start_b)

    # - (x < 0) trick to adjust for negative numbers( in case length is 0)
    end_a = int(start_a + a[4] - 1 + 0.5 - bool_to_float((a[4] - 1) < 0))
    end_b = int(start_b + b[4] - 1 + 0.5 - bool_to_float((b[4] - 1) < 0))
    end = min(min(end_a, end_b), N_OFFSETS - 1)
    # if (end < start) return 1e9;
    if (end < start):
        return False

    # dist = 0
    # for i in range(5 + start, 5 + end + 1):
    #     # print( 'a[i] - b[i]: ',a[i] - b[i])
    #     dist += (b[i] - a[i]) if a[i] < b[i] else (a[i] - b[i])

    diff_ab = a[5 + start:5 + end + 1] - b[5 + start:5 + end + 1]
    dist = torch.sum(torch.abs(diff_ab))

    return dist < (threshold * (end - start + 1))

def devIoUs(a, b, threshold: float = 0.5):
    start_a = int(a[2] * N_STRIPS - DATASET_OFFSET + 0.5)  # 0.5 rounding trick
    start_b = (b[:,2] * N_STRIPS - DATASET_OFFSET + 0.5).detach().cpu().numpy().astype(np.int32)
    start = np.maximum(start_a, start_b)

    # - (x < 0) trick to adjust for negative numbers( in case length is 0)
    end_a = int(start_a + a[4] - 1 + 0.5 - bool_to_float((a[4] - 1) < 0))
    end_b = (start_b+(b[:,4] - 1 + 0.5).detach().cpu().numpy()+((b[:, 4] - 1) < 0).detach().cpu().numpy().astype(np.float32)).astype(np.int32)
    end = np.minimum(N_OFFSETS - 1, np.minimum(end_a, end_b))

    T, C = b.size()
    a = a.unsqueeze(0).repeat(T,1)

    mask=np.zeros((T,C))
    thr_dis=np.zeros((T))
    for n in range(T):
        f=start[n]
        l=end[n]
        mask[n,f:l]=1
        thr_dis[n]=(l - f + 1)
    mask = torch.from_numpy(mask).to(b.device)
    thr_dis=torch.from_numpy(thr_dis*threshold).to(b.device)

    diff_ab =a-b
    dist = diff_ab*mask
    dist = torch.sum(torch.abs(dist),dim=1)

    idxs = torch.where(dist<thr_dis)[0]

    return idxs.detach().cpu().numpy()

def nms_forward_cpu(boxes, scores, overlap: float = 0.0, top_k: int = 3000):
    # return [], 0 ,None
    sort_score, index = scores.sort(0, True)

    # print(f'score: {sort_score}, index {index}')
    index = index.detach().cpu().numpy().tolist()

    keep = []
    num_to_keep=0
    while len(index) > 0:

        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        num_to_keep+=1
        if num_to_keep >= top_k:
            break

        un_over_index=[]
        for n in index[1:]:
            if devIoU(boxes[i], boxes[n],overlap):
                un_over_index.append(n)

        index = un_over_index

    return keep, num_to_keep, None

def nms_forward_speedup(boxes, scores, overlap: float = 0.0, top_k: int = 3000):
    boxes=boxes.cpu()

    sort_score, index = scores.sort(0, True)

    index = index.detach().cpu().numpy()

    keep = []
    num_to_keep = 0
    while len(index) > 0:

        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        num_to_keep += 1
        if num_to_keep >= top_k:
            break

        un_overlap = devIoUs(boxes[i], boxes[index[1:]], overlap)

        index = index[un_overlap + 1]

    return keep, num_to_keep, None


nms_forward = nms_forward_speedup