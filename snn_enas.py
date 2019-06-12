import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import permutations

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, baseWidth, stride=1, downsample=None, separate_coef=1):
        super(Bottleneck, self).__init__()
        D = int(planes * (baseWidth / (16*separate_coef)))##
        C = cardinality*2
        C_group = 4*2
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, groups=separate_coef, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C_group, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, groups=separate_coef, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, y=None):
        
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
        
        if residual.size() != out.size():
            print(out.size(), residual.size())
        out += residual
        out = self.relu(out)
        
        return out


class ResNeXt_Cifar(nn.Module):

    def __init__(self, block, layers, cardinality, baseWidth, num_classes=10, is_separate=True):
        super(ResNeXt_Cifar, self).__init__()
        self.cardinality = cardinality
        self.separate_coef = cardinality if is_separate else 1
        self.inplanes = 16*self.separate_coef #64##
        self.baseWidth = baseWidth
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)##
        self.bn1 = nn.BatchNorm2d(16)##
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.separate_coef, layers[0])##
        self.layer2 = self._make_layer(block, 16*self.separate_coef*2, layers[1], stride=2)##
        self.layer3 = self._make_layer(block, 16*self.separate_coef*4, layers[2], stride=2)##
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)##
        
        self.decision = list(permutations([0,1,2,3]))
        
        
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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, groups=self.separate_coef, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth, stride, downsample, separate_coef=self.separate_coef))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth, separate_coef=self.separate_coef))
        
        return nn.Sequential(*layers)

    def forward(self, x, policy=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.separate_coef>1:
            x = torch.cat([x for _ in range(self.cardinality)], 1)
            device_num = self.cardinality
            s = 0
            for stage in [self.layer1, self.layer2, self.layer3]:
                y = None
                for i, layer in enumerate(list(stage.modules())[0]):
                    x = layer(x)
                    if i%2==0:
                        if policy is not None:
                            chunk = torch.split(x, x.size(1)//device_num, 1)
                            d = self.decision[policy[s]]
                            a = torch.cat([chunk[k] for k in d], 1)
                        else:
                            print("no policy...")
                            shift = x.shape[1]//device_num*(s%(device_num-1)+1)
                            a = torch.cat([x[:,shift:], x[:,:shift]], 1)
                        s += 1
                    elif i%2==1:
                        x = (x+a)/2

            start = 0*x.shape[1]//device_num
            end = 1*x.shape[1]//device_num
            x = x[:,start:end]
        
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resneXt_cifar(depth, cardinality, baseWidth, **kwargs):
    assert (depth - 2) % 9 == 0
    n = (depth - 2) // 9
    model = ResNeXt_Cifar(Bottleneck, [n, n, n], cardinality, baseWidth, **kwargs)
    return model


