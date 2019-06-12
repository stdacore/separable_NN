import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import time

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, baseWidth, stride=1, downsample=None, separate_coef=1):
        super(Bottleneck, self).__init__()
        D = int(planes * (baseWidth / (16*separate_coef)))##
        C = cardinality
        C_group = 4
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, groups=separate_coef, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C_group, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, groups=separate_coef, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, y=None, policy=None):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
#         shift = out.shape[1]//4*(y%(4-1)+1)
#         y = torch.cat([out[:,shift:], out[:,:shift]], 1)
#         out = (out+y)/2
        
        
#         if y is None:
#             y = out
#         else:
#             if policy is not None:
#                 out = (out+out*(1-policy)+y*policy)/2
#             else:
#                 out = (out+y)/2
#             y = None
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if residual.size() != out.size():
            print(out.size(), residual.size())
        
        out += residual
        out = self.relu(out)
        
        
#         p = torch.zeros(out.shape[0], out.shape[1]).cuda()
#         p = p+policy.view(-1, 1)
#         p = p.view(out.shape[0], out.shape[1], 1, 1)
#         out = out*p + residual*(1-p)
        
        
        return out#, y


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
            for k, stage in enumerate([self.layer1, self.layer2, self.layer3]):
                y = None
                for i, layer in enumerate(list(stage.modules())[0]):
                    
#                     x = layer(x, s, policy[:,s])
# #                     if y is not None:
# #                         shift = y.shape[1]//device_num*(s%(device_num-1)+1)
# #                         y = torch.cat([y[:,shift:], y[:,:shift]], 1)
# #                         if policy is not None:
# #                             p = torch.zeros(y.shape[0], y.shape[1]).cuda()
# #                             p = p+policy[:,3*k+i//2].view(-1, 1)
# #                             p = p.view(y.shape[0], y.shape[1], 1, 1)
# # #                             p = torch.ones(y.shape[0], y.shape[1], 1, 1).cuda()
#                     s += 1

                    
                    
                    
                    
                    x = layer(x)
                    if i%2==0:
                        shift = x.shape[1]//device_num*(s%(device_num-1)+1)
                        a = torch.cat([x[:,shift:], x[:,:shift]], 1)
                        if policy is not None:
                            p = torch.zeros(x.shape[0], x.shape[1]//device_num//2).cuda()
                            p = p+policy[:,6*k+i].view(-1, 1)
                            q = torch.zeros(x.shape[0], x.shape[1]//device_num//2).cuda()
                            q = q+policy[:,6*k+i+1].view(-1, 1)
                            p = torch.cat([p, q], 1)
#                             p = torch.tensor([[i for i in range(x.shape[1]//device_num)] for j in range(x.shape[0])]).cuda()
#                             p = p/(x.shape[1]//device_num)
#                             po = (policy[:,i].view(-1, 1)*2+policy[:,i+9].view(-1, 1))*0.34
#                             p = torch.floor(p.to(torch.float)+po)
#                             print(p)
                            
                            p = torch.cat([p for _ in range(device_num)], 1)
                            p = p.view(x.shape[0], x.shape[1], 1, 1)
#                             p = torch.ones(x.shape[0], x.shape[1], 1, 1).cuda()
                            a = a*p
                        s += 1
                    elif i%2==1:
                        x = (x+x*(1-p)+a*p)/2
            

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


class FlatResNet(nn.Module):

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy):

        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                action = policy[:,t].contiguous()
                residual = self.ds[segment](x) if b==0 else x

                # early termination if all actions in the batch are zero
                if action.data.sum() == 0:
                    x = residual
                    t += 1
                    continue

                action_mask = action.float().view(-1,1,1,1)
                fx = F.relu(residual + self.blocks[segment][b](x))
                x = fx*action_mask + residual*(1-action_mask)
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                if policy[t]==1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def forward_full(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# Smaller Flattened Resnet, tailored for CIFAR
class FlatResNet32(FlatResNet):

    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

#---------------------------------------------------------------------------------------------------------#

# Class to generate resnetNB or any other config (default is 3B)
# removed the fc layer so it serves as a feature extractor
class Policy32(nn.Module):

    def __init__(self, layer_config=[1,1,1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32(BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)


    def forward(self, x):
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value
