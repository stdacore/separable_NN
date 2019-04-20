import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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
#         print(x.shape)
        if self.conv1.weight.shape[1]!=x.shape[1]:
            p = nn.AvgPool2d(2, stride=2)
            x = p(x)
            x = torch.cat((x, x), 1)

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
        self.device_num = 2
        self.layer_num = layers[0]+layers[1]+layers[2]
        self.transmission_time = 0
        self.path = []
        
        for device in range(1, self.device_num+1):
            self.inplanes = 16
            tmp = []
#             for layer in range(1, self.layer_num+1):
            tmp.extend(self._make_layer(block, 16, layers[0], device=device, layer=1))
            tmp.extend(self._make_layer(block, 32, layers[1], stride=2, device=device, layer=2))
            tmp.extend(self._make_layer(block, 64, layers[2], stride=2, device=device, layer=3))
            self.path.append(tmp)
        
#         tmp = []
#         layer1 = self._make_layer(block, 16, layers[0])
#         layer2 = self._make_layer(block, 32, layers[1], stride=2)
#         layer3 = self._make_layer(block, 64, layers[2], stride=2)
#         tmp.extend(layer1)
#         tmp.extend(layer2)
#         tmp.extend(layer3)
#         self.path = [tmp for _ in range(self.device_num)]
#         print(self.path)
        
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, device=1, layer=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        setattr(self, 'layer_%d_block_%d_device_%d'%(layer, 0, device), block(self.inplanes, planes, stride, downsample))
        layers.append(getattr(self, 'layer_%d_block_%d_device_%d'%(layer, 0, device)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            setattr(self, 'layer_%d_block_%d_device_%d'%(layer, i, device), block(self.inplanes, planes))
            layers.append(getattr(self, 'layer_%d_block_%d_device_%d'%(layer, i, device)))
        
        return layers
#         return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        flow = [x for _ in range(self.device_num)]
        send = [[None, 0, True] for _ in range(self.device_num)]
#         a = 0
#         b = 0
#         while a+1<self.layer_num and b+1<self.layer_num:
#             print(a)
#             print(b)
#             if (a+b)%2==0:
#                 flow[0] = self.path[0][a+1](self.path[0][a](flow[0]))
#                 flow[1] = self.path[1][b](flow[1])
#                 a = a+2
#                 b = b+1
#                 t = flow[1]
#                 if flow[0].shape!=flow[1].shape:
#                     p = nn.AvgPool2d(2, stride=2)
#                     t = p(t)
#                     t = torch.cat((t, t), 1)
#                 flow[0] = (flow[0]+t)/2
#             else:
#                 flow[1] = self.path[1][b+1](self.path[1][b](flow[1]))
#                 flow[0] = self.path[0][a](flow[1])
#                 a = a+1
#                 b = b+2
#                 t = flow[0]
#                 if flow[0].shape!=flow[1].shape:
#                     p = nn.AvgPool2d(2, stride=2)
#                     t = p(t)
#                     t = torch.cat((t, t), 1)
#                 flow[1] = (flow[1]+t)/2
            
        
        
#         flow[0] = x
#         for i in range(self.layer_num):
#             tmp = []
#             for j in range(self.device_num):
                
#                 if flow[j] is not None:
#                     flow[j] = self.path[j][i](flow[j])
#                 else:
#                     flow[j] = self.path[j][i](x)
                
#                 from_device_idx = (j+(i%(self.device_num-1)+1))%self.device_num
#                 s = send[from_device_idx]
                
#                 if s is not None and s[0] is not None and s[1]<i-self.transmission_time and not s[2]:
# #                 if s is not None and s[0] is not None and not s[2]:
# #                     print(i, j, from_device_idx)
#                     if flow[j] is None:
#                         flow[j] = s[0]
#                     else:
#                         if s[0].shape!=flow[j].shape:
#                             p = nn.AvgPool2d(2, stride=2)
#                             s[0] = p(s[0])
#                             s[0] = torch.cat((s[0], s[0]), 1)
# #                         s[0] = torch.zeros(s[0].shape).cuda('cuda:1')
#                         flow[j] = flow[j] + s[0]
#                         flow[j] = flow[j]/2
#                     send[from_device_idx][2] = True
                
#                 if flow[j] is not None and send[j][2]:
# #                 tmp.append([flow[j], i, False])
# #             for j in range(self.device_num):
#                     send[j]=[flow[j], i, False]
        
        
        y = flow[0]
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)

        return y
    
    
def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model

def sresnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [9, 9, 9], **kwargs)
    return model
