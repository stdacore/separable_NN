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
from snn_graph_3 import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--split', default=1, type=int, help='number of devices to split the model')
parser.add_argument('--epoch', default=200, type=int, help='epoch')
parser.add_argument('--batch', default=128, type=int, help='batch')
parser.add_argument('--schedule', default=50, type=int, help='schedule to decay learning rate')
parser.add_argument('--cuda', default=0, type=int, help='gpu index')
parser.add_argument('--resume', '-r', default='', type=str, help='checkpoint path to resume')
parser.add_argument('--save', default='exp', type=str, help='checkpoint path to save')
parser.add_argument('--penalty', type=float, default=-0.1, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--cl_step', type=int, default=1, help='steps for curriculum training')

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

torch.backends.cudnn.benchmark = True
device = 'cuda'#'cuda:%d'%args.cuda #
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
#     CIFAR10Policy(),
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
net = resneXt_cifar(56, 4, 16, num_classes=100, is_separate=False)
net = net.to(device)

# print(net)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

agent = Policy32([1,1,1], num_blocks=18)
agent = agent.to(device)

# print(agent)
if args.resume:

    print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/resnext_56_4_16_171.t7')
    #resnext_56_4_16_compress_7_198.t7
    #resnext_56_4_16_config1_170.t7
    #resnext_56_4_16_config7_190.t7
    #resnext_56_4_16_compress_5_168.t7
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
net = nn.DataParallel(net)
agent = nn.DataParallel(agent)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(agent.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=0)
num_blocks = 18

def get_reward(preds, targets, policy):

    block_use = policy.sum(1).float()/policy.size(1)
    sparse_reward = 1.0-block_use**2

    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data

    reward = sparse_reward
    reward[1-match] = args.penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
#     net.train()
    agent.train()
    net.eval()
    matches, rewards, policies = [], [], []
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        probs, value = agent(inputs)
        
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = torch.distributions.Bernoulli(probs)
        policy = distr.sample()

        if args.cl_step < num_blocks:
            policy[:, :-args.cl_step] = 1
            policy_map[:, :-args.cl_step] = 1

            policy_mask = Variable(torch.ones(inputs.size(0), policy.size(1))).cuda()
            policy_mask[:, :-args.cl_step] = 0
        else:
            policy_mask = None
        
        with torch.no_grad():
            v_inputs = Variable(inputs.data, requires_grad=False)
            preds_map = net(v_inputs, policy_map)
            preds_sample = net(v_inputs, policy)

        reward_map, _ = get_reward(preds_map, targets, policy_map.data)
        reward_sample, match = get_reward(preds_sample, targets, policy.data)

        advantage = reward_sample - reward_map

        loss = -distr.log_prob(policy)
        loss = loss * Variable(advantage).expand_as(policy)

        if policy_mask is not None:
            loss = policy_mask * loss # mask for curriculum learning

        loss = loss.sum()

        probs = probs.clamp(1e-15, 1-1e-15)
        entropy_loss = -probs*torch.log(probs)
        entropy_loss = 0.1*entropy_loss.sum()

        loss = (loss - entropy_loss)/inputs.size(0)

        #---------------------------------------------------------------------#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        correct += match.sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        policies.append(policy.data)
        
#     print(policy_map.data[0])
#     print(policy.data[:10])
#     print(reward_map)
#     print(reward_sample)
    policies = torch.cat(policies, 0)
#     print(policies)
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()
    print(policies.sum(0))
    print(sparsity.item(), variance.item())
    state = {
                'net': agent.state_dict(),
                'epoch': epoch,
            }
    torch.save(state, './checkpoint/agent.t7')

def test(epoch):
    global best_acc
    agent.eval()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    matches, rewards, policies = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            probs, _ = agent(inputs)

            policy = probs.data.clone()
            policy[policy<0.5] = 0.0
            policy[policy>=0.5] = 1.0
            policy = Variable(policy)
            
            if args.cl_step < num_blocks:
#                 policy_keep = policy.data.clone()
                policy[:, :-args.cl_step] = 1
            
            preds= net(inputs, policy)
            reward, match = get_reward(preds, targets, policy.data)
            
            total += targets.size(0)
            correct += match.sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (0/(batch_idx+1), 100.*correct/total, correct, total))
            
            policies.append(policy.data)
    
#     print(a[0])
    policies = torch.cat(policies, 0)
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()
    print(policies.sum(0))
    print(sparsity.item(), variance.item())
#         print(p.data[0])    
#         print(policy.data[0])

        
for epoch in range(args.epoch):
    
    train(epoch)
    if epoch%10==0:
        
        test(epoch)
    if args.cl_step < num_blocks:
        args.cl_step = 1 + 1 * (epoch // 1)
    else:
        args.cl_step = num_blocks

#     schedule = args.schedule
    
#     if epoch in [schedule, schedule*2, schedule*3]:
#         optimizer.param_groups[0]['lr'] /= 10

#     print('now learning rate is %.4f'%optimizer.param_groups[0]['lr'])
    

#python3 main.py --lr 0.1 --split 1 --epoch 200 --batch 128 --schedule 50 --cuda 1 --save resnext_56_2_16