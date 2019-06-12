'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

import os, sys, random
import argparse

from autoaugment import CIFAR10Policy

from snn_enas import *
from utils import progress_bar
from controller import Controller

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--split', default=1, type=int, help='number of devices to split the model')
parser.add_argument('--epoch', default=200, type=int, help='epoch')
parser.add_argument('--batch', default=128, type=int, help='batch')
parser.add_argument('--schedule', default=50, type=int, help='schedule to decay learning rate')
parser.add_argument('--cstep', default=500, type=int, help='controller update steps per epoch')
parser.add_argument('--cuda', default=-1, type=int, help='gpu index')
parser.add_argument('--resume', '-r', default='', type=str, help='checkpoint path to resume')
parser.add_argument('--save', '-s', default='exp', type=str, help='checkpoint path to save')
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
if args.cuda==-1:
    device = 'cuda'
else:
    device = 'cuda:%d'%args.cuda #'cuda'#

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainset, validset = torch.utils.data.random_split(trainset, [len(trainset)-int(0.1*len(trainset)), int(0.1*len(trainset))])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=10)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=10)


# Model
print('==> Building model..')
net = resneXt_cifar(110, 4, 16, num_classes=100, is_separate=True)
print(net)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

if args.cuda==-1:
    net = nn.DataParallel(net)
net = net.to(device)

controller = Controller(args)
print(controller)
print(sum(p.numel() for p in controller.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
shared_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
controller_optimizer = optim.Adam(controller.parameters(), lr=0.001, weight_decay=5e-4)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s'%args.resume)#resnext_56_4_16_config1_170.t7')#

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# if args.cuda==-1:
#     net = nn.DataParallel(net)
# net = net.to(device)

    
# if args.cuda==-1:
#     controller = nn.DataParallel(controller)
controller = controller.to(device)


# Training
def train_shared(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    controller.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        policy = controller.sample()
        inputs, targets = inputs.to(device), targets.to(device)
        shared_optimizer.zero_grad()
        outputs = net(inputs, policy)
        loss = criterion(outputs, targets)
        loss.backward()
        shared_optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | %s'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, policy))
    log_value('train_loss', train_loss/(batch_idx+1), epoch)
    log_value('train_accuracy', correct/total, epoch)

def train_controller(epoch):
    controller.train()
    avg_reward_base = None
    baseline = None
    adv_history = []
    entropy_history = []
    reward_history = []
    
    
    total_loss = 0
    controller_step = 0
    
    while controller_step<args.cstep:
        for batch_idx, (inputs, targets) in enumerate(validloader):
    #     for step in range(self.args.controller_max_step):
            # sample models
            if controller_step>=args.cstep:
                break
            policy, log_probs, entropies = controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with torch.no_grad():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs, policy)
                _, predicted = outputs.max(1)
                rewards = predicted.eq(targets).sum().item()/inputs.size(0)

            reward_history.append(rewards)
            entropy_history.append(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = 0.95#self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.append(adv)

            # policy loss
            loss = -log_probs*torch.tensor(adv).to(device)
    #         if self.args.entropy_mode == 'regularizer':
    #             loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            controller_optimizer.zero_grad()
            loss.backward()
            controller_optimizer.step()

            total_loss += loss.item()

#             if ((step % self.args.log_step) == 0) and (step > 0):
#                 self._summarize_controller_train(total_loss,
#                                                  adv_history,
#                                                  entropy_history,
#                                                  reward_history,
#                                                  avg_reward_base,
#                                                  dags)

#                 reward_history, adv_history, entropy_history = [], [], []
#                 total_loss = 0

            progress_bar(controller_step, args.cstep, 'Adv: %.3f | Reward: %.3f | %s'% (sum(adv_history)/(controller_step+1), sum(reward_history)/(controller_step+1), policy))
            controller_step += 1
        
    log_value('train_adv', sum(adv_history)/controller_step, epoch)
    log_value('train_reward', sum(reward_history)/controller_step, epoch)

def test(epoch, best=0):
#     net.eval()
#     controller.eval()
    correct = 0
    total = 0
    sample_num = 0
    sample_log = []
    
    with torch.no_grad():
        while sample_num<100:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if sample_num>=100:
                    break
                policy = controller.sample(is_train=False)
                inputs, targets = inputs.to(device), targets.to(device)
                shared_optimizer.zero_grad()
                outputs = net(inputs, policy)

                _, predicted = outputs.max(1)
                total = targets.size(0)
                correct = predicted.eq(targets).sum().item()

                progress_bar(sample_num, 100, 'Acc: %.3f%% (%d/%d) | %s' % (100.*correct/total, correct, total, policy))

                sample_log.append(correct/total)
                sample_num+=1
                
                if correct/total>best:
                    state = {
                        'net': net.state_dict(),
                        'controller': controller.state_dict(),
                        'policy': policy,
                        'correct': correct/total,
                        'epoch': epoch,
                    }
                    best = correct/total
                    
    log_value('test_accuracy', best, epoch)
    sample_log = torch.tensor(sample_log)
    log_value('test_accuracy_mean', sample_log.mean().item(), epoch)
    log_value('test_accuracy_std', sample_log.std().item(), epoch)
    
    print(state['correct'], state['policy'])
    torch.save(state, './checkpoint/%s_%d_%.3f.t7'%(args.save, epoch, best))

                
                
# best = 0
configure('cv/'+args.save+'/log', flush_secs=5)
for epoch in range(args.epoch):
    
    train_shared(epoch)
    train_controller(epoch)
    test(epoch)
    schedule = args.schedule
    
#     if epoch in [schedule, schedule*2, schedule*3]:
#         shared_optimizer.param_groups[0]['lr'] /= 10

#     print('now learning rate is %.4f'%optimizer.param_groups[0]['lr'])
    

