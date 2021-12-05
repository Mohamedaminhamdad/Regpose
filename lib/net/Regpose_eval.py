import os
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms
import numpy as np

"""
    This Function is only used during test 
    achnors are filtered out using Nonlinear Maximum Supression in order to be left with only the true bounding boxes. 
"""
class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes
def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out
class Regpose(nn.Module):
    def __init__(self,arg, backbone, bifpn, head,Rot_head,Tran_head):
        super(Regpose, self).__init__()
        self.backbone = backbone
        self.arg=arg
        self.bifpn = bifpn
        self.head = head
        self.rot=Rot_head
        self.tran=Tran_head
        self.threshold=0.2 # score threshold as it was implemented in Efficientdet
        self.iou_threshold=0.2 # Iou Threshold for bounding boxes  implemented in Efficientdet

    def forward(self, x,L):     
        """
        x: Input Image Features [bs,3,480,640]
        features: 
        """                
        P1,P2,P3,P4,P5=self.backbone(x) 
        inputs=[P1,P2,P3,P4,P5] # Output Backbone Network : shape [[bs,64,120,160],[bs,64,120,160],[bs,128,60,80],[bs,256,30,40],[bs,512,15,20]
        features=self.bifpn(inputs) #Output BiFPN: [P3_out:[bs,256,120,160],P4_out:[bs,256,120,160],P5_out:[bs,256,60,80],P6_out:[bs,256,30,40],P7_ou:[bs,256,15,20]]
        anchors,regressor, classes=self.head(features,x) # Output: anchors: bbox prios, regressor: Offset-anchor gt bbox, class probability map for each anchor
        if not self.arg.pytorch.gt and self.arg.pytorch.test: 
            regressBoxes = BBoxTransform() 
            clipBoxes = ClipBoxes()
            output=postprocess(x, anchors, regressor, classes, regressBoxes, clipBoxes, self.threshold, self.iou_threshold)
            bbox=output[0]['rois']
            cls=output[0]['class_ids']
            u=0
            k,_=bbox.shape
            bbox=np.expand_dims(bbox,0)
            bbox_roi=torch.cat((u*torch.ones(1,k,1),torch.from_numpy(bbox)),2)
            if self.arg.pytorch.gpu > -1:
                bbox_roi=bbox_roi.squeeze(0).cuda()
            else:
                bbox_roi=bbox_roi.squeeze(0)
            q=self.rot(features,bbox_roi)
            t=self.tran(features,bbox_roi)
            return cls,bbox,q,t
        elif self.arg.pytorch.gt and self.arg.pytorch.test:
            q=self.rot(features,L)
            t=self.tran(features,L)
            return q,t
        if not self.arg.pytorch.test:
            q=self.rot(features,L)
            t=self.tran(features,L)      
            return  anchors,regressor, classes,q,t
        