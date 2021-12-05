import torch.nn as nn
import math
import torch
from torch import nn
import torch.nn.functional as F
from net.utils import Swish, SeparableConvBlock,MaxPool2dStaticSamePadding
class BiFPN(nn.Module):

    """
    Initial Implementation from: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    Modified to work with Resnet-34
    """

    def __init__(self, num_channels, freeze=False,epsilon=1e-4, onnx_export=False, first=True):
        """

        Args:
            num_channels:
            conv_channels:
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.first=first
        self.freeze=freeze
        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
  

        # Feature scaling layers
        self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=1, mode='nearest')

        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.convP12=nn.Conv2d(64,256,kernel_size=1,stride=1,padding=0,bias=False)
        self.convP3=nn.Conv2d(128,256,kernel_size=1,stride=1,padding=0,bias=False)
        self.convP5=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=False)
        # Batch Normalitation for every Input Feature Map
        self.bn1=nn.BatchNorm2d(256)
        self.bn2=nn.BatchNorm2d(256)
        self.bn3=nn.BatchNorm2d(256)
        self.bn4=nn.BatchNorm2d(256)
        self.swish =  Swish()
        # Weight For the BiFON 
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()


    def forward(self, inputs):


        return self._forward_fast_attention(inputs)
        
     

        

    def _forward_fast_attention(self, inputs):
        
        """
            Resize Resnet Inputs  to have  unified Channel Siize [256,w,h] using 1x1 conv
            Input: p3_in, p4_in, p5_in, p7_in = inputs with different Channel Size
            p6_in excluded from this, it has already [265,h,w]
            Output: P3_out, P4_out, P5_out, P6_out, P7_out Fast-Forward Fused Features: 
            Illustration of Futer Fusing BiFPN
            P7_0 -------------------------> P7_out -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_up ---------> P6_out -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_up ---------> P5_out -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_up ---------> P4_out -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_out -------->
        P6_up: Conv(w1 P_6_in +w2Resize(P7_in)/(w1+w2+epsilon))
        P6_out: Conv(w1' P_6_in +w_2' P6_up + w3'Resize(P5_out)/(w1'+w2'+w3'+epsilon))

        """
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        p3_in=self.convP12(p3_in)   # In: [bn,64,120,160] Out: [bn,256,120,160] 
        p3_in=self.bn1(p3_in)       # Perform Batch Normalization
        p4_in=self.convP12(p4_in)   # In: [bn,64,120,160] Out: [bn,256,120,160] 
        p4_in=self.bn2(p4_in)
        p5_in=self.convP3(p5_in)   # In: [bn, 128, 60, 80] Out: [bn,256,60,80] 
        p5_in=self.bn3(p5_in)
        p7_in=self.convP5(p7_in) # In: [bn,512,15,20] Out: [bn,256,15,20]
        p7_in=self.bn4(p7_in)     
        if self.freeze:
            with torch.no_grad():
                # Faster normalized Feature Fusion: P6_up: Conv(w1 P_6_in +w2Resize(P7_in)/(w1+w2+epsilon)) Intermediete node
                # Faster normalized Feature Fusion: P6_out: Conv(w1' P_6_in +w_2' P6_up + w3'Resize(P5_out)/(w1'+w2'+w3'+epsilon)) Output node
                # All weights are normalized
                p6_w1 = self.p6_w1_relu(self.p6_w1)
                weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
                # Connections for P6_in and P7_in to P6_up respectively
                p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p7_upsample(p7_in)))

                # Weights for P5_in and P6_up to P5_up
                p5_w1 = self.p5_w1_relu(self.p5_w1)
                weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
                # Connections for P5_in and P6_up to P5_up respectively
                p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p6_upsample(p6_up)))

                # Weights for P4_in and P5_up to P4_up
                p4_w1 = self.p4_w1_relu(self.p4_w1)
                weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
                # Connections for P4_in and P5_up to P4_up respectively
                p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p5_upsample(p5_up)))

                # Weights for P3_in and P4_up to P3_ou
                p3_w2 = self.p3_w2_relu(self.p3_w2)
                weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
                # Connections for P3_in and P4_up to P3_out respectively
                p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p4_upsample(p4_up)))

                # Weights for P4_in, P4_up and P3_out to P4_out
                p4_w2 = self.p4_w2_relu(self.p4_w2)
                weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
                # Connections for P4_in, P4_up and P3_out to P4_out respectively
                p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * p3_out))

                # Weights for P5_in, P5_in and P4_out to P5_out
                p5_w2 = self.p5_w2_relu(self.p5_w2)
                weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
                # Connections for P5_in, P5_up and P4_out to P5_out respectively
                p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p4_downsample(p4_out)))

                # Weights for P6_in, P6_up and P5_out to P6_out
                p6_w2 = self.p6_w2_relu(self.p6_w2)
                weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
                # Connections for P6_in, P6_up and P5_out to P6_out respectively
                p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p5_downsample(p5_out)))

                # Weights for P7_in and P6_out to P7_out
                p7_w2 = self.p7_w2_relu(self.p7_w2)
                weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
                # Connections for P7_in and P6_out to P7_out
                p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p6_downsample(p6_out)))
            return p3_out.detach(), p4_out.detach(), p5_out.detach(), p6_out.detach(), p7_out.detach()

        else: 
            # Weights for P6_in and P7_in to P6_up
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # Connections for P6_in and P7_in to P6_up respectively
            p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p7_upsample(p7_in)))

            # Weights for P5_in and P6_in to P5_up
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            # Connections for P5_in and P6_up to P5_up respectively
            p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p6_upsample(p6_up)))

            # Weights for P4_in and P5_up to P4_up
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            # Connections for P4_in and P5_up to P4_up respectively
            p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p5_upsample(p5_up)))

            # Weights for P3_in and P4_up to P3_out
            p3_w2 = self.p3_w2_relu(self.p3_w2)
            weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
            # Connections for P3_in and P4_up to P3_out respectively
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p4_upsample(p4_up)))

            # Weights for P4_in, P4_up and P3_out to P4_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            # Connections for P4_in, P4_up and P3_out to P4_out respectively
            p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * p3_out))
            # Weights for P5_in, P5_up and P4_out to P5_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            # Connections for P5_in, P5_up and P4_out to P5_out respectively
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p4_downsample(p4_out)))

            # Weights for P6_in, P6_up and P5_out to P6_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            # Connections for P6_in, P6_up and P5_out to P6_out respectively
            p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p5_downsample(p5_out)))

            # Weights for P7_in and P6_out to P7_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # Connections for P7_in and P6_out to P7_out
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p6_downsample(p6_out)))
            return p3_out, p4_out, p5_out, p6_out, p7_out

