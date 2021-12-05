import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys
from utils.utils import CustomDataParallel


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import utils.logger as logger
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

from torch.nn.modules import pooling
from net.BiFpn_net import BiFPN
from net.rot_net import Rot_Reg
from net.tran_net import Tran_Reg
import numpy as np 
from net.resnet_backbone import ResNetBackboneNet
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys
import cv2
from net.head import Head 
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import math
import torchvision
#from net.Regpose import Regpose
from net.Regpose_eval import Regpose
# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
def init_weigths(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)
        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0, std=0.001)
def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


# Re-init optimizer
def build_model(cfg):
    ## get model and optimizer
    res=[0.25,4,8,16,32]
    if 'resnet' in cfg.network.arch:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[cfg.network.back_layers_num]
        backbone_net = ResNetBackboneNet(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        if cfg.network.back_freeze:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, backbone_net.parameters()),
                                   'lr': float(cfg.train.lr)})
    Bi_net = BiFPN(256,freeze=cfg.network.class_head_freeze)
    head_net= Head(freeze=cfg.network.class_head_freeze)
    Rot_head=Rot_Reg(rot_rep=cfg.network.rot_representation,freeze=cfg.network.rot_head_freeze)
    Tran_head=Tran_Reg(freeze=cfg.network.trans_head_freeze)
    init_weigths(Bi_net)
    init_weigths(head_net)
    init_weigths(Rot_head)
    init_weigths(Tran_head)
    
    params_lr_list.append({'params': filter(lambda p: p.requires_grad, Bi_net.parameters()),
                                   'lr': float(cfg.train.lr)})
    params_lr_list.append({'params': filter(lambda p: p.requires_grad, head_net.parameters()),
                                   'lr': float(cfg.train.lr)})
    params_lr_list.append({'params': filter(lambda p: p.requires_grad, Rot_head.parameters()),
                                   'lr': float(cfg.train.lr)})
    params_lr_list.append({'params': filter(lambda p: p.requires_grad, Tran_head.parameters()),
                                   'lr': float(cfg.train.lr)})
    if params_lr_list != []:
        if cfg.train.optimizer_name=="RMSProp":
            optimizer = torch.optim.RMSprop(params_lr_list, alpha=cfg.train.alpha, eps=float(cfg.train.epsilon),
                                            weight_decay=cfg.train.weightDecay, momentum=cfg.train.momentum)
        elif cfg.train.optimizer_name=="adamw":
            optimizer = torch.optim.AdamW(params_lr_list,float(cfg.train.lr),betas=(cfg.train.momentum, cfg.train.Beta),eps=float(cfg.train.epsilon), weight_decay=cfg.train.weightDecay )
    else:
        optimizer = None
    
    model = Regpose(cfg,backbone_net, Bi_net, head_net,Rot_head,Tran_head)
    ## model initialization
    if cfg.pytorch.load_model != '':
        logger.info("=> loading model '{}'".format(cfg.pytorch.load_model))
        #pretrained_dict=torch.load(cfg.pytorch.load_model)
        #model_dict=model.state_dict()
        #pretrained_dict={k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)
        model.load_state_dict(torch.load(cfg.pytorch.load_model,map_location=torch.device('cpu')),strict=False)
    else:
        if 'resnet' in cfg.network.arch:
            logger.info("=> loading official model from model zoo for backbone")
            _, _, _, name = resnet_spec[cfg.network.back_layers_num]
            #official_resnet = model_zoo.load_url(model_urls[name])
            # drop original resnet fc layer, add 'None' in case of no fc layer, that will raise error
            official_resnet=torch.load('resnet34.pth')
            #model.load_state_dict('resnet34.pth')
            official_resnet.pop('fc.weight', None)
            official_resnet.pop('fc.bias', None)
            model.backbone.load_state_dict(official_resnet)
    return model, optimizer



def save_model(model,path):
    if isinstance(model,CustomDataParallel):
        torch.save(model.module.state_dict(),path)
    else:
        torch.save(model.state_dict(),path)

