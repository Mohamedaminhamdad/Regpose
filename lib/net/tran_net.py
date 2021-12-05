import torch 
import torch.nn as nn
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from net.utils import SeparableConvBlock
class Tran_Reg(nn.Module):
    def __init__(self,epsilon=1e-4,freeze=False):
        super(Tran_Reg, self).__init__()
        self.epsilon=epsilon
        self.freeze=freeze
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lin1 = nn.ModuleList()
        self.lin1.append(nn.Linear(4000, 4000))
        self.lin1.append(nn.ReLU(inplace=True))
        self.lin1.append(nn.Linear(4000, 4000))
        self.lin1.append(nn.ReLU(inplace=True))
        self.lin1.append(nn.Linear(4000, 3))
        self.p4_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_weight_relu = nn.ReLU()
        self.p3_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_weight_relu = nn.ReLU()
        self.p2_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_weight_relu = nn.ReLU()
        self.relu=nn.ReLU()
        num_filters=64
        num_layers=3
        self.features=nn.ModuleList()
        for i in range(num_layers):
            _in_channels = 256 if i == 0 else num_filters
            self.features.append(SeparableConvBlock(_in_channels, num_filters))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.LeakyReLU(inplace=True))
        self.deconv=nn.ModuleList()
        self.deconv.append(torch.nn.ConvTranspose2d(num_filters, num_filters, 4, stride=4, padding=1, output_padding=2, groups=1, dilation=1))
        self.deconv.append(nn.BatchNorm2d(num_filters))
        self.deconv.append(nn.LeakyReLU(inplace=True))
        self.conv_up1 = SeparableConvBlock(256)
        self.conv_up2 = SeparableConvBlock(256,64)
        self.conv_up4 = SeparableConvBlock(256,64)
        self.conv_up3=SeparableConvBlock(64,10)
        self.deconv1=torch.nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=2, groups=1, dilation=1)
        self.x1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.x2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.x3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=1, mode='nearest')
    def forward(self,x,bbox): 
        if self.freeze:
            with torch.no_grad():
                p2_w1 = self.p2_weight_relu(self.p2_weight)
                weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
                P1 = self.conv_up1(self.relu(weight[0] * x[3] + weight[1] *self.x1_upsample(x[4])))
                for i, l in enumerate(self.features):
                    P1=l(P1)
                for i, l in enumerate(self.deconv):
                    P1=l(P1)
                p3_w1 = self.p3_weight_relu(self.p3_weight)
                weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
                P2 = self.conv_up2(self.relu(weight[0] * x[1]+ weight[1] * self.x2_upsample(x[2])))
                P3=self.conv_up4(x[0])
                p4_w1 = self.p4_weight_relu(self.p4_weight)
                weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
                p3_up = self.conv_up3(self.relu(weight[0] * P1 + weight[1] *P2 +weight[2]*P3 )) # Perform Feature Fusion 
                P= torchvision.ops.roi_pool(p3_up,bbox,20,1/4)  # Perform Roi Pooling
                P = P.view(-1, 10*20*20)
                for i, l in enumerate(self.lin1):
                    P = l(P)
                P= F.normalize(P,dim=1,p=2)
                return P.detach()
        else:
            p2_w1 = self.p2_weight_relu(self.p2_weight)
            weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
            P1 = self.conv_up1(self.relu(weight[0] * x[3] + weight[1] *self.x1_upsample(x[4])))
            for i, l in enumerate(self.features):
                        P1=l(P1)
            for i, l in enumerate(self.deconv):
                P1=l(P1)
            p3_w1 = self.p3_weight_relu(self.p3_weight)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            P2 = self.conv_up2(self.relu(weight[0] * x[1]+ weight[1] * self.x2_upsample(x[2])))
            P3=self.conv_up4(x[0])
            p4_w1 = self.p4_weight_relu(self.p4_weight)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p3_up = self.conv_up3(self.relu(weight[0] * P1 + weight[1] *P2 +weight[2]*P3 )) # Feature Fusion of all three Feature maps
            P= torchvision.ops.roi_pool(p3_up,bbox,20,1/4)  # Perform Roi Pooling
            P = P.view(-1, 10*20*20)
            for i, l in enumerate(self.lin1):
                P = l(P)
        return P
