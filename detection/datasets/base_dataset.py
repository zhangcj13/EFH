import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
# from detection.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC
import random
import copy

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None,
            cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Process(processes, cfg) if processes is not None else None


    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            # imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)
    
    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {self._imgpath % img_id} not found"

        return img
    
    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img
    
    def pull_item(self,index):
        data_info = self.data_infos[index]
        # sample = data_info.copy()
        sample =copy.deepcopy(data_info)
        if 'img_path' in sample.keys():
            img = cv2.imread(data_info['img_path'], cv2.IMREAD_COLOR)
            assert img is not None, f"file named {data_info['img_path']} not found"
            sample.update({'img': img})
        return sample
    
    def mosaic_item(self,idx):
        input_h, input_w = [self.cfg.img_h,self.cfg.img_w]
        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        mosaic_bbox=[]
        mosaic_sample=None
        for i_mosaic, index in enumerate(indices):
            sample = self.pull_item(index)
            img = sample['img']
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_sample = sample
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            if 'boxes' in sample.keys():
                for b in sample['boxes']:
                    b.scale_pad(scale,padw, padh)
                    b.clip(0,0, 2 * input_w, 2 * input_h)
                mosaic_bbox.extend(sample['boxes'])
        
        mosaic_sample['img']=mosaic_img
        mosaic_sample['boxes']=mosaic_bbox
        
        return mosaic_sample
                
    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})
        
        return sample 
    
    def mixup(self, sample, label='boxes'):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        len_labels = 0
        while len_labels == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            data_info = self.data_infos[cp_index]
            len_labels = len(data_info[label])
            
        cp_sample = self.pull_item(cp_index)
        img=cp_sample['img']

        if len(img.shape) == 3:
            cp_img = np.ones((self.cfg.img_h, self.cfg.img_w, 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones((self.cfg.img_h, self.cfg.img_w), dtype=np.uint8) * 114

        cp_scale_ratio = min(self.cfg.img_h / img.shape[0], self.cfg.img_w / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
