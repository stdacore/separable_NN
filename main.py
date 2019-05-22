'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os, sys, random
import argparse

from autoaugment import CIFAR10Policy

# from seperable_net import *
from snn_graph import *
from snn_graph_2 import *
from snn_graph_3 import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--split', default=1, type=int, help='number of devices to split the model')
parser.add_argument('--epoch', default=200, type=int, help='epoch')
parser.add_argument('--batch', default=128, type=int, help='batch')
parser.add_argument('--schedule', default=50, type=int, help='schedule to decay learning rate')
parser.add_argument('--cuda', default=0, type=int, help='gpu index')
parser.add_argument('--resume', '-r', default='', type=str, help='checkpoint path to resume')
parser.add_argument('--save', default='exp', type=str, help='checkpoint path to save')
args = parser.parse_args()

print('===== parameter settings =====')
print('learining rate: %.4f'%args.lr)
print('split: %d'%args.split)
print('epoch: %d'%args.epoch)
print('schedule: %d'%args.schedule)
print('cuda: %d'%args.cuda)
print('resume: %s'%args.resume)
print('save: %s'%args.save)
print('===== parameter settings =====')


device = 'cuda'#'cuda:%d'%args.cuda 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(5),
    CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=10)


# Model
print('==> Building model..')
# net = sresnet164_cifar(num_classes=100)
net = resneXt_cifar(110, 4, 16, num_classes=100, is_separate=True)
# net = densenet_BC_cifar(250, 24, num_classes=100)
net = nn.DataParallel(net)
net = net.to(device)
# net2 = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
# net2 = net2.to(device)
# net3 = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
# net3 = net3.to(device)
# net4 = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
# net4 = net4.to(device)
# net5 = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
# net5 = net4.to(device)

# for p in net.parameters():
#     print(p.nelement())
print(net)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

if args.resume:

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/%s'%args.resume)
    checkpoint = torch.load('./checkpoint/resnext_56_1_16_167.t7')
    checkpoint2 = torch.load('./checkpoint/resnext_56_1_16_re2_198.t7')
    checkpoint5 = torch.load('./checkpoint/resnext_56_1_16_re5_157.t7')
    checkpoint3 = torch.load('./checkpoint/resnext_56_1_16_re3_155.t7')
    checkpoint4 = torch.load('./checkpoint/resnext_56_1_16_re4_159.t7')
    
#     for key in checkpoint['net'].keys():
#         if 'layer' in key and 'num_batches_tracked' not in key:
#             checkpoint['net'][key] = torch.cat([checkpoint['net'][key], checkpoint2['net'][key], checkpoint3['net'][key], checkpoint4['net'][key]], 0)
#             x = checkpoint['net'][key]
#             shape = x.shape
#             shape[0] = shape[0]//4
#             r = random.randint(0, shape[0]*3)
#             checkpoint['net'][key] = torch.cat([x, x, x, x], 0)
#             tensor = torch.randn(checkpoint['net'][key].shape)/50
#             checkpoint['net'][key] = checkpoint['net'][key].to('cpu')+tensor
#         substring = key.split('device_')
#         from_key = substring[0]+'device_1'+substring[1][1:] if len(substring)>=2 else key
#         if key!=from_key:
#             checkpoint['net'][key] = checkpoint['net'][from_key]


    net.load_state_dict(checkpoint['net'])
    net2.load_state_dict(checkpoint2['net'])
    net3.load_state_dict(checkpoint3['net'])
    net4.load_state_dict(checkpoint4['net'])
    net5.load_state_dict(checkpoint5['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# print(checkpoint['net'].keys())
# sys.exit()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
#     net2.eval()
#     net3.eval()
#     net4.eval()
#     net5.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
#             outputs = (net(inputs)+net2(inputs)+net3(inputs)+net5(inputs))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        if epoch > args.schedule*2:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            net.to('cpu')
            torch.save(state, './checkpoint/%s_%d.t7'%(args.save, epoch))
            net.to(device)
        best_acc = acc
        
for epoch in range(args.epoch):
    
    train(epoch)
    test(epoch)
    schedule = args.schedule
    
    if epoch in [schedule, schedule*2, schedule*3]:
        optimizer.param_groups[0]['lr'] /= 10

    print('now learning rate is %.4f'%optimizer.param_groups[0]['lr'])
    

#python3 main.py --lr 0.1 --split 1 --epoch 200 --batch 128 --schedule 50 --cuda 1 --save resnext_56_2_16