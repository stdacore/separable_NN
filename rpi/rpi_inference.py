'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from autoaugment import CIFAR10Policy

# from densenet_cifar import *
#from resnet_cifar import *
# from seperable_net import *
from mobilenetv2 import *
from utils import progress_bar

device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

# Model
print('==> Building model..')
# net = preact_resnet164_cifar()
# net = densenet_BC_cifar(depth=100, k=12, num_classes=100)
net = MobileNetV2()
net = net.to(device)

# for p in net.parameters():
#     print(p.nelement())
# print(net)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))


# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
print('==> Resuming from checkpoint..')
#assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./dense100c100_190_21.91_cpu.t7')
# net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']

class WrappedModel(nn.Module):
    def __init__(self, net):
        super(WrappedModel, self).__init__()
        self.module = net # that I actually define.
    def forward(self, x):
        return self.module(x)

# then I load the weights I save from previous code:
net = WrappedModel(net)
checkpoint = torch.load('./mobilenetv2c10_194_cpu.t7')
net.load_state_dict(checkpoint['net'], strict=False)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    best_acc = acc
    
test(0)    
