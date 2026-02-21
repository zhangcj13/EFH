import torch

import torch.nn.functional as F

from ..registry import HEADS
# from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss


@HEADS.register_module
class YOLOX_Head(YOLOXHead):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        cfg=None) -> None:
        super().__init__(num_classes, width,strides,in_channels,act,depthwise)

        self.cfg=cfg
        self.head_num=len(self.strides)
        self.forawd_yolox=cfg.forawd_yolox if hasattr(cfg,'forawd_yolox') else False

    def forward(self, features,labels=None, imgs=None, **kwargs):
        if self.forawd_yolox:
            return YOLOXHead.forward(self,features,labels, imgs)

        assert len(features)>=self.head_num
        xin = features[-self.head_num:]

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)
        # if self.training:
        #     return self.get_losses(
        #         imgs,
        #         x_shifts,
        #         y_shifts,
        #         expanded_strides,
        #         labels,
        #         torch.cat(outputs, 1),
        #         origin_preds,
        #         dtype=xin[0].dtype,
        #     )
        # else:
        #     self.hw = [x.shape[-2:] for x in outputs]
        #     # [batch, n_anchors_all, 85]
        #     outputs = torch.cat(
        #         [x.flatten(start_dim=2) for x in outputs], dim=2
        #     ).permute(0, 2, 1)
        #     if self.decode_in_inference:
        #         return self.decode_outputs(outputs, dtype=xin[0].type())
        #     else:
        #         return outputs
        
        results = {'pred_boxes': outputs,
                   'x_shifts':x_shifts,
                   'y_shifts':y_shifts,
                   'expanded_strides':expanded_strides,
                   'origin_preds':origin_preds,}
        return results
        
    def loss(self, output, batch):
        total_loss,loss_iou,loss_obj,loss_cls,loss_l1,fg_per=self.get_losses(
                                                                        batch['img'],
                                                                        output['x_shifts'],
                                                                        output['y_shifts'],
                                                                        output['expanded_strides'],
                                                                        batch['bounding_box'],
                                                                        torch.cat(output['pred_boxes'], 1),
                                                                        output['origin_preds'],
                                                                        dtype=output['pred_boxes'][0].dtype,)
        
        return {'bbox_loss': total_loss, 'loss_stats': {'bbox_loss': total_loss,
                                                        'loss_iou':loss_iou,
                                                        'loss_obj':loss_obj,
                                                        'loss_cls':loss_cls,
                                                        'loss_l1':loss_l1,
                                                        'fg_percent':fg_per}}
    
    def get_bboxes(self,output):
        pred_boxes = output['pred_boxes']
        self.hw = [x.shape[-2:] for x in pred_boxes]
        # [batch, n_anchors_all, 85]
        pred_boxes = torch.cat(
            [x.flatten(start_dim=2) for x in pred_boxes], dim=2
        ).permute(0, 2, 1)

        prediction=self.decode_outputs(pred_boxes, dtype=pred_boxes[0].type())
        if self.cfg.ori_img_h is None or self.cfg.ori_img_w is None:
            height,width =[self.cfg.img_h, self.cfg.img_w]
        else:
            height,width =[self.cfg.ori_img_h, self.cfg.ori_img_w]
        
        predictions = postprocess_network_output(prediction, self.num_classes,
                                                 height=height, 
                                                 width=width)

        batch_boxes = []
        for result in predictions:
            boxes=[]
            for box,score,label in zip(result['boxes'],result['scores'],result['labels']):
                left, top, right, bottom=box.detach().cpu().numpy()

                left  = left/self.cfg.img_w*width
                right = right/self.cfg.img_w*width
                top   = top/self.cfg.img_h*height
                bottom= bottom/self.cfg.img_h*height

                predicted_class = int(label.detach().cpu().numpy())
                score = score.detach().cpu().numpy()
                boxes.append({'box': [left, top, right, bottom], 'class': predicted_class, 'score': score})
            
            batch_boxes.append(boxes)

        return batch_boxes
    
    def infer_snn(self, xin):
        outputs = []
 
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)

        outputs=[torch.mean(x, dim=-1, keepdim=False) for x in outputs]

        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].type())
        else:
            return outputs
        
import torchvision
def batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold, width, height):
    # adopted from torchvision nms, but faster
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_dim = max([width, height])
    offsets = idxs * float(max_dim + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep

def postprocess_network_output(prediction, num_classes, conf_thre=0.01, nms_thre=0.65, height=640, width=640, filtering=True,class_agnostic=False):
    prediction[..., :2] -= prediction[...,2:4] / 2 # cxcywh->xywh
    prediction[..., 2:4] += prediction[...,:2] # xywh->xyxy

    output = []
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if len(image_pred) == 0:
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float),
                "labels": torch.zeros(0, dtype=torch.long)
            })
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        image_pred[:, 4:5] *= class_conf

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # detections = torch.cat((image_pred[:, :5], class_pred), 1)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)

        if filtering:
            detections = detections[conf_mask]

        if len(detections) == 0:
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float),
                "labels": torch.zeros(0, dtype=torch.long)
            })
            continue

        # nms_out_index = batched_nms_coordinate_trick(detections[:, :4], detections[:, 4], detections[:, 5],
        #                                               nms_thre, width=width, height=height)
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        if filtering:
            detections = detections[nms_out_index]

        output.append({
            "boxes": detections[:, :4],
            "scores": detections[:, 4]*detections[:, 5],
            "labels": detections[:, 6].long()
        })

    return output

if __name__ == '__main__':
    head=YOLOX_Head(num_classes=2)
    print(head)