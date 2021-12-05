import torchvision.models as models
import director
import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys
#from torchsummary import summary
from utils.utils import CustomDataParallel


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import utils.logger as logger
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

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
def build_model(arg):
    ## get model and optimizer
    res=[0.25,4,8,16,32]
    if 'resnet' in arg.network.arch:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[34]
    backbone_net = ResNetBackboneNet(block_type, layers, arg.network.back_input_channel, arg.network.back_freeze)
    Bi_net = BiFPN(256,freeze=arg.network.class_head_freeze)
    head_net= Head(freeze=arg.network.class_head_freeze)
    Rot_head=Rot_Reg(rot_rep=arg.network.rot_representation,freeze=arg.network.rot_head_freeze)
    Tran_head=Tran_Reg(freeze=arg.network.trans_head_freeze)
    
    model = Regpose(arg,backbone_net, Bi_net, head_net,Rot_head,Tran_head)
    model.load_state_dict(torch.load(arg.pytorch.load_model,map_location=torch.device('cpu')),strict=False)


    return model



def save_model(model,path):
    if isinstance(model,CustomDataParallel):
        torch.save(model.module.state_dict(),path)
    else:
        torch.save(model.state_dict(),path)

