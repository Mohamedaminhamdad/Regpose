import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
import itertools
from net.utils import Swish, SeparableConvBlock
class Head(nn.Module):
    def __init__(self, freeze=False,num_classes=21):
        super(Head, self).__init__()
        self.fpn_num_filters = 256
        self.box_class_repeats = 3 # Nmber of convolutional Layers
        self.pyramid_levels = 5 # Number of pyramid_levels 
        self.anchor_scale = 4 
        self.freeze=freeze  # Freeze or Unfreeze NEtwork
        self.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.num_scales = len( [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) 
        self.num_anchors = len(self.aspect_ratios) * self.num_scales

        self.regressor = Regressor(in_channels=self.fpn_num_filters, num_anchors=self.num_anchors,
                                   num_layers=self.box_class_repeats,
                                   pyramid_levels=self.pyramid_levels,freeze=self.freeze)
        self.anchors = Anchors(anchor_scale=self.anchor_scale,
                               pyramid_levels=(torch.arange(self.pyramid_levels) + 1).tolist())
        self.classifier = Classifier(in_channels=self.fpn_num_filters, num_anchors=self.num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats,
                                     pyramid_levels=self.pyramid_levels,freeze=self.freeze)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if  m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,feat,x):
        regressor = self.regressor(feat)
        anchors=self.anchors(x)   
        classes=self.classifier(feat)
        return anchors,regressor,classes

class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5,freeze=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.freeze=freeze
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish =  Swish()

    def forward(self, inputs):
        if self.freeze:
            with torch.no_grad():
                feats = []
                for feat, bn_list in zip(inputs, self.bn_list):
                    for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                        feat = conv(feat)
                        feat = bn(feat)
                        feat = self.swish(feat)
                    feat = self.header(feat)

                    feat = feat.permute(0, 2, 3, 1)
                    feat = feat.contiguous().view(feat.shape[0], -1, 4)

                    feats.append(feat)

                feats = torch.cat(feats, dim=1)

            return feats.detach()
        else:
            feats = []
            for feat, bn_list in zip(inputs, self.bn_list):
                for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                    feat = conv(feat)
                    feat = bn(feat)
                    feat = self.swish(feat)
                feat = self.header(feat)

                feat = feat.permute(0, 2, 3, 1)
                feat = feat.contiguous().view(feat.shape[0], -1, 4)

                feats.append(feat)

            feats = torch.cat(feats, dim=1)
            return feats
    



class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by  https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

    Modified by Mohamed
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = pyramid_levels
        self.strides = [4,4,8,16,32]
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
class Classifier(nn.Module):
    """
    modified by     
            https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

    Modified by Mohamed
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, freeze=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.freeze=freeze
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish =  Swish()

    def forward(self, inputs):
        if self.freeze:
            with torch.no_grad():
                feats = []
                for feat, bn_list in zip(inputs, self.bn_list):
                    for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                        feat = conv(feat)
                        feat = bn(feat)
                        feat = self.swish(feat)
                    feat = self.header(feat)

                    feat = feat.permute(0, 2, 3, 1)
                    feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
                    feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

                    feats.append(feat)

                feats = torch.cat(feats, dim=1)
                feats = feats.sigmoid()
                return feats.detach()
        
        else:
            feats = []
            for feat, bn_list in zip(inputs, self.bn_list):
                for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                    feat = conv(feat)
                    feat = bn(feat)
                    feat = self.swish(feat)
                feat = self.header(feat)

                feat = feat.permute(0, 2, 3, 1)
                feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
                feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

                feats.append(feat)

            feats = torch.cat(feats, dim=1)
            feats = feats.sigmoid()

            return feats
