import torch 
import torch.nn as nn
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from net.utils import SeparableConvBlock
def R_from_6d(Output_network):
    m1 = Output_network[:,0:3] # Network gives an Output of 6 degree of freedom. three of them are useless. 
    m2 = Output_network[:,3:6]
    """
    In order to recover the rotaiton matric from the 6dof representation the Gram-Schmit process is used: normalization followed
    by cross products to build a rotation matrix
    See: https://arxiv.org/abs/1812.07035
    """    
    x = norm(m1)
    z = cross_product(x,m2) 
    z = norm(z)
    y = cross_product(z,x)   
    matrix = torch.cat((x.view(-1,3,1),y.view(-1,3,1),z.view(-1,3,1)), 2) # Rotation Matrix lying in the SO(3) 
    return matrix
def norm(x):
    x_abs = torch.sqrt(x.pow(2).sum(1))
    if torch.cuda.is_available():
        x_abs = torch.max(x_abs, torch.autograd.Variable(torch.FloatTensor([1e-8])).cuda())
    else: 
        x_abs = torch.max(x_abs, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    x_abs = x_abs.view(x.shape[0],1).expand(x.shape[0],v.shape[1])
    x_norm = x/x_abs
    return x_norm

def cross_product( x, y):
    p1 = x[:,1]*y[:,2] - x[:,2]*y[:,1]
    p2 = x[:,2]*y[:,0] - x[:,0]*y[:,2]
    p3 = x[:,0]*y[:,1] - x[:,1]*y[:,0]
    cross= torch.cat((p1.view(u.shape[0],1), p2.view(u.shape[0],1), p3.view(u.shape[0],1)),1)    
    return cross
class Rot_Reg(nn.Module):
    """
    Rotation Regression Head to regress the 6DoF Pose of an Object. 
    Input: List containing Features: [[bs,256,120,160],[bs,256,120,160],[bs,256,60,80],[bs,256,30,40],[bs,256,15,20]]
    Output: [bn,rotatation] quat: [bs,4]  | 6D: [bs,6]
    ILlustration of The Feature FUsion in Rotation Net Pi_out is the output of BiFPN
            P7_out -----------
                             |
                             ↓                
            P6_out ---------> P1 ---------> Conv---> Conv--->Conv--->Deconv----------------------- 
                                                                                                  |
            P5_out -----------                                                                    |                                                                                                                                           
                             |                                                                    |
                             ↓                                                                    ↓ 
            P4_out ---------> P2 ---------------------------------------------------------->  Pinp_roi -----> Roi-Pooling ----> FC-Layers
                                                                                                
                                                                                                  |
            P4_out ---------------------------------> Conv -----------------------------------> P3

    """
    def __init__(self,rot_rep,epsilon=1e-4,freeze=False):
        super(Rot_Reg, self).__init__()
        self.rot_rep=rot_rep
        self.epsilon=epsilon
        self.freeze=freeze
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lin1 = nn.ModuleList()  # Fully Connected Block 
        self.lin1.append(nn.Linear(4000, 4000))
        self.lin1.append(nn.ReLU(inplace=True))
        self.lin1.append(nn.Linear(4000, 4000))
        self.lin1.append(nn.ReLU(inplace=True))
        if self.rot_rep=='quat':
            self.lin1.append(nn.Linear(4000, 4))
        else: 
            self.lin1.append(nn.Linear(4000, 6))
        self.p4_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) # Fusion Parameter for Feature Fusion
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
        self.conv_up4=SeparableConvBlock(256,64)
        self.conv_up3=SeparableConvBlock(64,10)
        self.x1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.x2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.x3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=1, mode='nearest')
    def forward(self,x,bbox):  ## INput : BiFPN Features and extracted bBounding Boxes 
        if self.freeze:
            with torch.no_grad():
                p2_w1 = self.p2_weight_relu(self.p2_weight)  # Weight Inialization for Feature Fusion between P7_out and P6_out 
                weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)  # normalization 
                P1 = self.conv_up1(self.relu(weight[0] * x[3] + weight[1] *self.x1_upsample(x[4])))  # P1=conv(w1*P6_out+w2 Resize(P7_out)/w1+w2+epsilon)
                for i, l in enumerate(self.features): # Convolutional Operation 
                    P1=l(P1)
                for i, l in enumerate(self.deconv): #Deconv of fused Features 
                    P1=l(P1)
                p3_w1 = self.p3_weight_relu(self.p3_weight) # Weight Inialization for Feature Fusion between P5_out and P4_out 
                weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon) # normalization 
                P2 = self.conv_up2(self.relu(weight[0] * x[1]+ weight[1] * self.x2_upsample(x[2]))) # P2=conv(w1*P4_out+w2 Resize(P5_out)/w1+w2+epsilon)
                P3=self.conv_up4(x[0])  # Convolutional Operation  For P3_out 
                p4_w1 = self.p4_weight_relu(self.p4_weight) ## Weight Initialization for featur fusion between P1 P2 and P3
                weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
                pinp_roi = self.conv_up3(self.relu(weight[0] * P1 + weight[1] *P2 +weight[2]*P3 ))  # Feature Fusion of P1 P2 and P3
                P= torchvision.ops.roi_pool(pinp_roi,bbox,20,1/4)   # Roi Pooling  
                P = P.view(-1, 10*20*20) # Sequence of Fully COnnected Layers
                for i, l in enumerate(self.lin1):
                    P = l(P)
                if self.rot_rep=='quat':
                    P= F.normalize(P,dim=1,p=2)
                else: 
                    P= R_from_6d(P)  # [bs,6]
                return P.detach()
        else:
            p2_w1 = self.p2_weight_relu(self.p2_weight) # Weight Inialization for Feature Fusion between P7_out and P6_out 
            weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon) # normalization 
            P1 = self.conv_up1(self.relu(weight[0] * x[3] + weight[1] *self.x1_upsample(x[4]))) # P1=conv(w1*P6_out+w2 Resize(P7_out)/w1+w2+epsilon)
            for i, l in enumerate(self.features):   # Convolutional Operation 
                        P1=l(P1)
            for i, l in enumerate(self.deconv):  # Deconv of FUsed Features 
                P1=l(P1)
            p3_w1 = self.p3_weight_relu(self.p3_weight) # Weight Inialization for Feature Fusion between P5_out and P4_out 
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon) # normalization 
            P2 = self.conv_up2(self.relu(weight[0] * x[1]+ weight[1] * self.x2_upsample(x[2]))) # P2=conv(w1*P4_out+w2 Resize(P5_out)/w1+w2+epsilon)
            P3=self.conv_up4(x[0])    # Convolutional Operation  For P3_out 
            p4_w1 = self.p4_weight_relu(self.p4_weight)  ## Weight Initialization for featur fusion between P1 P2 and P3
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            pinp_roi = self.conv_up3(self.relu(weight[0] * P1 + weight[1] *P2 +weight[2]*P3 )) # Feature Fusion
            P= torchvision.ops.roi_pool(pinp_roi,bbox,20,0.25)  # Roi Pooling 
            P = P.view(-1, 10*20*20)
            for i, l in enumerate(self.lin1):
                P = l(P)
            if self.rot_rep=='quat':
                P= F.normalize(P,dim=1,p=2)
            else: 
                P= R_from_6d(P)  # [bs,6]
        return P
