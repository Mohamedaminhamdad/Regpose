# _from trochvision Resnet block https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch.nn as nn
import torch

class ResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False):
        self.freeze = freeze
        self.inplanes = 64
        super(ResNetBackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._create_block(block, 64, layers[0])
        self.layer2 = self._create_block(block, 128, layers[1], stride=2)
        self.layer3 = self._create_block(block, 256, layers[2], stride=2)
        self.layer4 = self._create_block(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _create_block(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # Input: [bs,3,480,640]
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)   # First convolutional Layer in Resnet-Arch [bs, 64, 240, 320]
                x = self.bn1(x)
                x = self.relu(x)
                x1 = self.maxpool(x)  # After maxpooling: [bs, 64, 120, 160]
                x2 = self.layer1(x1)  # Output Residual Block 1:  [bs, 64, 120, 160]
                x3 = self.layer2(x2)  # Output Residual Block 2: [bs, 128, 60, 80]
                x4 = self.layer3(x3)  # Output Residual Block 3:  [bs, 256, 30, 40]
                x5 = self.layer4(x4)  # Output Residual Block 4:  [bn, 512, 15, 20]
                return x1.detach(),x2.detach(),x3.detach(),x4.detach(),x5.detach()
        else:
            x = self.conv1(x)    # First convolutional Layer in Resnet-Arch [bs, 64, 240, 320]
            x = self.bn1(x)
            x = self.relu(x)
            x1 = self.maxpool(x)  # After maxpooling: [bs, 64, 120, 160]
            x2 = self.layer1(x1)  # Output Residual Block 1:  [bs, 64, 120, 160]
            x3 = self.layer2(x2)  # Output Residual Block 2: [bs, 128, 60, 80]
            x4 = self.layer3(x3)  # Output Residual Block 3:  [bs, 256, 30, 40]
            x5 = self.layer4(x4)  # Output Residual Block 4:  [bs, 512, 15, 20]
            return x1,x2,x3,x4,x5
