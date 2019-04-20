import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out
    
    
class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
#         ### case 1 ###
#         self.layer1 = self._make_layer(block, 16, layers[0])
#         self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
#         ### end ###

#         ### case 2 ###
#         self.layer1a = self._make_layer(block, 16, blocks=12, stride=1)
#         self.layer2a = self._make_layer(block, 32, blocks=6, stride=2)
#         self.layer3a = self._make_layer(block, 64, blocks=12, stride=2)
        
#         self.inplanes = 16
#         self.layer1b = self._make_layer(block, 16, blocks=6, stride=1)
#         self.layer2b = self._make_layer(block, 32, blocks=12, stride=2)
#         self.layer3b = self._make_layer(block, 64, blocks=6, stride=2)
#         ### end ###
        
        ### case 3 ###
        self.layer1a1 = self._make_layer(block, 16, blocks=2, stride=1)
        self.layer1a2 = self._make_layer(block, 16, blocks=1, stride=1)
        self.layer1a3 = self._make_layer(block, 16, blocks=2, stride=1)
        self.layer1a4 = self._make_layer(block, 16, blocks=1, stride=1)
        self.layer1a5 = self._make_layer(block, 16, blocks=2, stride=1)
        self.layer1a6 = self._make_layer(block, 16, blocks=1, stride=1)
        
        self.layer2a1 = self._make_layer(block, 32, blocks=2, stride=2)
        self.layer2a2 = self._make_layer(block, 32, blocks=1, stride=1)
        self.layer2a3 = self._make_layer(block, 32, blocks=2, stride=1)
        self.layer2a4 = self._make_layer(block, 32, blocks=1, stride=1)
        self.layer2a5 = self._make_layer(block, 32, blocks=2, stride=1)
        self.layer2a6 = self._make_layer(block, 32, blocks=1, stride=1)
        
        self.layer3a1 = self._make_layer(block, 64, blocks=2, stride=2)
        self.layer3a2 = self._make_layer(block, 64, blocks=1, stride=1)
        self.layer3a3 = self._make_layer(block, 64, blocks=2, stride=1)
        self.layer3a4 = self._make_layer(block, 64, blocks=1, stride=1)
        self.layer3a5 = self._make_layer(block, 64, blocks=2, stride=1)
        self.layer3a6 = self._make_layer(block, 64, blocks=1, stride=1)
        
        
        self.inplanes = 16
        self.layer1b1 = self._make_layer(block, 16, blocks=1, stride=1)
        self.layer1b2 = self._make_layer(block, 16, blocks=2, stride=1)
        self.layer1b3 = self._make_layer(block, 16, blocks=1, stride=1)
        self.layer1b4 = self._make_layer(block, 16, blocks=2, stride=1)
        self.layer1b5 = self._make_layer(block, 16, blocks=1, stride=1)
        self.layer1b6 = self._make_layer(block, 16, blocks=2, stride=1)
        
        self.layer2b1 = self._make_layer(block, 32, blocks=1, stride=2)
        self.layer2b2 = self._make_layer(block, 32, blocks=2, stride=1)
        self.layer2b3 = self._make_layer(block, 32, blocks=1, stride=1)
        self.layer2b4 = self._make_layer(block, 32, blocks=2, stride=1)
        self.layer2b5 = self._make_layer(block, 32, blocks=1, stride=1)
        self.layer2b6 = self._make_layer(block, 32, blocks=2, stride=1)
        
        self.layer3b1 = self._make_layer(block, 64, blocks=1, stride=2)
        self.layer3b2 = self._make_layer(block, 64, blocks=2, stride=1)
        self.layer3b3 = self._make_layer(block, 64, blocks=1, stride=1)
        self.layer3b4 = self._make_layer(block, 64, blocks=2, stride=1)
        self.layer3b5 = self._make_layer(block, 64, blocks=1, stride=1)
        self.layer3b6 = self._make_layer(block, 64, blocks=2, stride=1)
        ### end ###

        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
        
#         y = self.layer1b(x)
#         x = self.layer1a(x)
#         x = x+y
        
#         x = self.layer2a(x)
#         y = self.layer2b(y)
#         y = x+y

#         y = self.layer3b(y)
#         x = self.layer3a(x)
#         x = x+y    
    
    
        x = self.layer1a1(x) ### trap
        y = self.layer1b1(x)
        x = (x+y)/2
        
        x = self.layer1a2(x)
        y = self.layer1b2(y)
        y = (x+y)/2
        
        x = self.layer1a3(x)
        y = self.layer1b3(y)
        x = (x+y)/2
        
        x = self.layer1a4(x)
        y = self.layer1b4(y)
        y = (x+y)/2
        
        x = self.layer1a5(x)
        y = self.layer1b5(y)
        x = (x+y)/2
        
        x = self.layer1a6(x)
        y = self.layer1b6(y)
        y = (x+y)/2
    
        x = self.layer2a1(x)
        y = self.layer2b1(y)
        x = (x+y)/2
        
        x = self.layer2a2(x)
        y = self.layer2b2(y)
        y = (x+y)/2
        
        x = self.layer2a3(x)
        y = self.layer2b3(y)
        x = (x+y)/2
        
        x = self.layer2a4(x)
        y = self.layer2b4(y)
        y = (x+y)/2
        
        x = self.layer2a5(x)
        y = self.layer2b5(y)
        x = (x+y)/2
        
        x = self.layer2a6(x)
        y = self.layer2b6(y)
        y = (x+y)/2
        
        x = self.layer3a1(x)
        y = self.layer3b1(y)
        x = (x+y)/2
        
        x = self.layer3a2(x)
        y = self.layer3b2(y)
        y = (x+y)/2
        
        x = self.layer3a3(x)
        y = self.layer3b3(y)
        x = (x+y)/2
        
        x = self.layer3a4(x)
        y = self.layer3b4(y)
        y = (x+y)/2
        
        x = self.layer3a5(x)
        y = self.layer3b5(y)
        x = (x+y)/2
        
        x = self.layer3a6(x)
        y = self.layer3b6(y)
        y = (x+y)/2

#         x = self.bn(x)
        x = self.bn(y)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out    

class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
#         self.layer1a = self._make_layer(block, 6, layers[0])
#         self.layer2a = self._make_layer(block, 11, layers[1], stride=2)
#         self.layer3a = self._make_layer(block, 22, layers[2], stride=2)
        
#         self.inplanes = 16
#         self.layer1b = self._make_layer(block, 6, layers[0])
#         self.layer2b = self._make_layer(block, 11, layers[1], stride=2)
#         self.layer3b = self._make_layer(block, 22, layers[2], stride=2)
        
#         self.inplanes = 16
#         self.layer1c = self._make_layer(block, 6, layers[0])
#         self.layer2c = self._make_layer(block, 11, layers[1], stride=2)
#         self.layer3c = self._make_layer(block, 22, layers[2], stride=2)
        
        ### case 2 ###
#         self.layer1a = self._make_layer(block, 16, blocks=12, stride=1)
#         self.layer2a = self._make_layer(block, 32, blocks=6, stride=2)
#         self.layer3a = self._make_layer(block, 64, blocks=12, stride=2)
        
#         self.inplanes = 16
#         self.layer1b = self._make_layer(block, 16, blocks=6, stride=1)
#         self.layer2b = self._make_layer(block, 32, blocks=12, stride=2)
#         self.layer3b = self._make_layer(block, 64, blocks=6, stride=2)
        ### end ###

#         ### case 3 ###
#         self.layer1a1 = self._make_layer(block, 16, blocks=2, stride=1)
#         self.layer1a2 = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer1a3 = self._make_layer(block, 16, blocks=2, stride=1)
#         self.layer1a4 = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer1a5 = self._make_layer(block, 16, blocks=2, stride=1)
#         self.layer1a6 = self._make_layer(block, 16, blocks=1, stride=1)
        
#         self.layer2a1 = self._make_layer(block, 32, blocks=2, stride=2)
#         self.layer2a2 = self._make_layer(block, 32, blocks=1, stride=1)
#         self.layer2a3 = self._make_layer(block, 32, blocks=2, stride=1)
#         self.layer2a4 = self._make_layer(block, 32, blocks=1, stride=1)
#         self.layer2a5 = self._make_layer(block, 32, blocks=2, stride=1)
#         self.layer2a6 = self._make_layer(block, 32, blocks=1, stride=1)
        
#         self.layer3a1 = self._make_layer(block, 64, blocks=2, stride=2)
#         self.layer3a2 = self._make_layer(block, 64, blocks=1, stride=1)
#         self.layer3a3 = self._make_layer(block, 64, blocks=2, stride=1)
#         self.layer3a4 = self._make_layer(block, 64, blocks=1, stride=1)
#         self.layer3a5 = self._make_layer(block, 64, blocks=2, stride=1)
#         self.layer3a6 = self._make_layer(block, 64, blocks=1, stride=1)
        
        
#         self.inplanes = 16
#         self.layer1b1 = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer1b2 = self._make_layer(block, 16, blocks=2, stride=1)
#         self.layer1b3 = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer1b4 = self._make_layer(block, 16, blocks=2, stride=1)
#         self.layer1b5 = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer1b6 = self._make_layer(block, 16, blocks=2, stride=1)
        
#         self.layer2b1 = self._make_layer(block, 32, blocks=1, stride=2)
#         self.layer2b2 = self._make_layer(block, 32, blocks=2, stride=1)
#         self.layer2b3 = self._make_layer(block, 32, blocks=1, stride=1)
#         self.layer2b4 = self._make_layer(block, 32, blocks=2, stride=1)
#         self.layer2b5 = self._make_layer(block, 32, blocks=1, stride=1)
#         self.layer2b6 = self._make_layer(block, 32, blocks=2, stride=1)
        
#         self.layer3b1 = self._make_layer(block, 64, blocks=1, stride=2)
#         self.layer3b2 = self._make_layer(block, 64, blocks=2, stride=1)
#         self.layer3b3 = self._make_layer(block, 64, blocks=1, stride=1)
#         self.layer3b4 = self._make_layer(block, 64, blocks=2, stride=1)
#         self.layer3b5 = self._make_layer(block, 64, blocks=1, stride=1)
#         self.layer3b6 = self._make_layer(block, 64, blocks=2, stride=1)
#         ### end ###
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes) # group

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        t = time.time()
        x = self.layer1(x)
#         t1 = time.time()
#         print('layer1')
#         print(t1-t)
        x = self.layer2(x)
#         t2 = time.time()
#         print('layer2')
#         print(t2-t1)
        x = self.layer3(x)
#         t3 = time.time()
#         print('layer3')
#         print(t3-t2)
        
#         y = self.layer1a(x)
#         y = self.layer2a(y)
#         y = self.layer3a(y)
        
#         z = self.layer1b(x)
#         z = self.layer2b(z)
#         z = self.layer3b(z)
        
#         x = self.layer1c(x)
#         x = self.layer2c(x)
#         x = self.layer3c(x)
        
#         x = torch.cat([x,y,z], 1)
        
#         y = self.layer1b(x)
#         x = self.layer1a(x)
#         x = x+y
        
#         x = self.layer2a(x)
#         y = self.layer2b(y)
#         y = x+y

#         y = self.layer3b(y)
#         x = self.layer3a(x)
#         x = x+y    

#         y = self.layer1b1(x) ### trap
#         x = self.layer1a1(x) ### trap
#         x = (x+y)/2
        
#         x = self.layer1a2(x)
#         y = self.layer1b2(y)
#         y = (x+y)/2
        
#         x = self.layer1a3(x)
#         y = self.layer1b3(y)
#         x = (x+y)/2
        
#         x = self.layer1a4(x)
#         y = self.layer1b4(y)
#         y = (x+y)/2
        
#         x = self.layer1a5(x)
#         y = self.layer1b5(y)
#         x = (x+y)/2
        
#         x = self.layer1a6(x)
#         y = self.layer1b6(y)
#         y = (x+y)/2
    
#         x = self.layer2a1(x)
#         y = self.layer2b1(y)
#         x = (x+y)/2
        
#         x = self.layer2a2(x)
#         y = self.layer2b2(y)
#         y = (x+y)/2
        
#         x = self.layer2a3(x)
#         y = self.layer2b3(y)
#         x = (x+y)/2
        
#         x = self.layer2a4(x)
#         y = self.layer2b4(y)
#         y = (x+y)/2
        
#         x = self.layer2a5(x)
#         y = self.layer2b5(y)
#         x = (x+y)/2
        
#         x = self.layer2a6(x)
#         y = self.layer2b6(y)
#         y = (x+y)/2
        
#         x = self.layer3a1(x)
#         y = self.layer3b1(y)
#         x = (x+y)/2
        
#         x = self.layer3a2(x)
#         y = self.layer3b2(y)
#         y = (x+y)/2
        
#         x = self.layer3a3(x)
#         y = self.layer3b3(y)
#         x = (x+y)/2
        
#         x = self.layer3a4(x)
#         y = self.layer3b4(y)
#         y = (x+y)/2
        
#         x = self.layer3a5(x)
#         y = self.layer3b5(y)
#         x = (x+y)/2
        
#         x = self.layer3a6(x)
#         y = self.layer3b6(y)
#         y = (x+y)/2

        x = self.avgpool(x)
#         x = self.avgpool(y)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x    
    
def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model
    
# def SNet(**kwargs):
#     model = CifarSEResNet(CifarSEBasicBlock, **kwargs)
#     return model


# class CifarSEBasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1):
#         super(CifarSEBasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
# #         self.se = SELayer(planes, reduction)
#         if inplanes != planes:
#             self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
#                                             nn.BatchNorm2d(planes))
#         else:
#             self.downsample = lambda x: x
#         self.stride = stride

#     def forward(self, x):
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
# #         out = self.se(out)

#         out += residual
#         out = self.relu(out)

#         return out
    
    


# class CifarSEResNet(nn.Module):
#     def __init__(self, block, n_size, num_classes=10):
#         super(CifarSEResNet, self).__init__()
#         self.inplane = 16
#         self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplane)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.layer1a = self._make_layer(block, 16, blocks=n_size, stride=1)
#         self.layer2a = self._make_layer(block, 32, blocks=1, stride=2)
#         self.layer3a = self._make_layer(block, 64, blocks=n_size, stride=2)
        
#         self.inplane = 16
#         self.layer1b = self._make_layer(block, 16, blocks=1, stride=1)
#         self.layer2b = self._make_layer(block, 32, blocks=n_size, stride=2)
#         self.layer3b = self._make_layer(block, 64, blocks=1, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(64, num_classes)
#         self.initialize()

#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride):
#         strides = [stride] + [1] * (blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inplane, planes, stride))
#             self.inplane = planes ############# trap

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

# #         x = self.layer1a(x)
# #         x = self.layer2a(x)
# #         x = self.layer3a(x)
        
# #         x = self.layer1a(x) + self.layer1b(x)
# #         x = self.layer2b(x) + self.layer2a(x)
# #         x = self.layer3a(x) + self.layer3b(x)
        
#         y = self.layer1b(x)
#         x = self.layer1a(x)
#         x = x+y
        
#         x = self.layer2a(x)
#         y = self.layer2b(y)
#         y = x+y
    
#         y = self.layer3b(y)
#         x = self.layer3a(x)
#         x = x+y       

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x
    
    

# class SNet(nn.Module):
#     def __init__(self):
#         super(SNet, self).__init__()
#         self.conv0 = nn.Conv2d(3, 16, 3, 1, 1)
#         self.bn0 = nn.BatchNorm2d(16)
        
#         self.conv11a = nn.Conv2d(16, 16, 3, 1, 1)
#         self.bn11a = nn.BatchNorm2d(16)
#         self.conv12a = nn.Conv2d(16, 16, 3, 1, 1)
#         self.bn12a = nn.BatchNorm2d(16)
#         self.conv13a = nn.Conv2d(16, 16, 3, 1, 1)
#         self.bn13a = nn.BatchNorm2d(16)
#         self.conv11b = nn.Conv2d(16, 16, 3, 1, 1)
#         self.bn11b = nn.BatchNorm2d(16)
        
#         self.conv21b = nn.Conv2d(16, 32, 3, 1, 1)
#         self.bn21b = nn.BatchNorm2d(32)
#         self.conv22b = nn.Conv2d(32, 32, 3, 1, 1)
#         self.bn22b = nn.BatchNorm2d(32)
#         self.conv23b = nn.Conv2d(32, 32, 3, 1, 1)
#         self.bn23b = nn.BatchNorm2d(32)
#         self.conv21a = nn.Conv2d(16, 32, 3, 1, 1)
#         self.bn21a = nn.BatchNorm2d(32)
        
#         self.conv31a = nn.Conv2d(32, 64, 3, 1, 1)
#         self.bn31a = nn.BatchNorm2d(64)
#         self.conv32a = nn.Conv2d(64, 64, 3, 1, 1)
#         self.bn32a = nn.BatchNorm2d(64)
#         self.conv33a = nn.Conv2d(64, 64, 3, 1, 1)
#         self.bn33a = nn.BatchNorm2d(64)
#         self.conv31b = nn.Conv2d(32, 64, 3, 1, 1)
#         self.bn31b = nn.BatchNorm2d(64)
        
#         self.conv41b = nn.Conv2d(64, 128, 3, 1, 1)
#         self.bn41b = nn.BatchNorm2d(128)
#         self.conv42b = nn.Conv2d(128, 128, 3, 1, 1)
#         self.bn42b = nn.BatchNorm2d(128)
#         self.conv43b = nn.Conv2d(128, 128, 3, 1, 1)
#         self.bn43b = nn.BatchNorm2d(128)
#         self.conv41a = nn.Conv2d(64, 128, 3, 1, 1)
#         self.bn41a = nn.BatchNorm2d(128)
        
#         self.conv51b = nn.Conv2d(128, 128, 3, 1, 1)
#         self.bn51b = nn.BatchNorm2d(128)
        
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(128, 10)
        
# #         self.conv1 = nn.Conv2d(3, 6, 5)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #         self.fc1   = nn.Linear(16*5*5, 120)
# #         self.fc2   = nn.Linear(120, 84)
# #         self.fc3   = nn.Linear(84, 10)

#     def forward(self, x):
        
#         y = F.relu(self.bn0(self.conv0(x)))
        
#         x = F.relu(self.bn11a(self.conv11a(y)))
#         x = F.relu(self.bn12a(self.conv12a(x)))
#         x = F.relu(self.bn13a(self.conv13a(x)))
#         y = F.relu(self.bn11b(self.conv11b(y)))
#         x += y
        
#         x = F.max_pool2d(x, 2)
#         y = F.max_pool2d(y, 2)
        
#         x = F.relu(self.bn21a(self.conv21a(x)))
#         y = F.relu(self.bn21b(self.conv21b(y)))
#         y = F.relu(self.bn22b(self.conv22b(y)))
#         y = F.relu(self.bn23b(self.conv23b(y)))
#         y += x
        
#         x = F.max_pool2d(x, 2)
#         y = F.max_pool2d(y, 2)
        
#         x = F.relu(self.bn31a(self.conv31a(x)))
#         x = F.relu(self.bn32a(self.conv32a(x)))
#         x = F.relu(self.bn33a(self.conv33a(x)))
#         y = F.relu(self.bn31b(self.conv31b(y)))
#         x += y
        
#         x = F.max_pool2d(x, 2)
#         y = F.max_pool2d(y, 2)
        
#         x = F.relu(self.bn41a(self.conv41a(x)))
#         y = F.relu(self.bn41b(self.conv41b(y)))
#         y = F.relu(self.bn42b(self.conv42b(y)))
#         y = F.relu(self.bn43b(self.conv43b(y)))
#         y += x
        
#         x = F.max_pool2d(x, 2)
#         y = F.max_pool2d(y, 2)
        
#         y = F.relu(self.bn51b(self.conv51b(y)))
        
#         out = self.avgpool(y)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return out