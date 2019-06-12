import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli
import torchvision
import torchvision.transforms as transforms
from snn_graph_3 import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from autoaugment import CIFAR10Policy

import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R110_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--cl_step', type=int, default=1, help='steps for curriculum training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-1, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def get_reward(preds, targets, policy):

    block_use = policy.sum(1).float()/policy.size(1)
    sparse_reward = 1.0-block_use**2

    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data

    reward = sparse_reward
    reward[1-match] = args.penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()


def train(epoch):

    agent.train()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        probs, value = agent(inputs)

        #---------------------------------------------------------------------#

        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        if args.cl_step < num_blocks:
            policy[:, :-args.cl_step] = 1
            policy_map[:, :-args.cl_step] = 1

            policy_mask = Variable(torch.ones(inputs.size(0), policy.size(1))).cuda()
            policy_mask[:, :-args.cl_step] = 0
        else:
            policy_mask = None
        
        with torch.no_grad():
            v_inputs = Variable(inputs.data)
            preds_map = rnet(v_inputs, policy_map)
            preds_sample = rnet(v_inputs, policy)

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
        entropy_loss = args.beta*entropy_loss.sum()

        loss = (loss - entropy_loss)/inputs.size(0)

        #---------------------------------------------------------------------#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        policies.append(policy.data.cpu())

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'E: %d | A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, accuracy, reward, sparsity, variance, len(policy_set))
    print (log_str)

    log_value('train_accuracy', accuracy, epoch)
    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_unique_policies', len(policy_set), epoch)


def test(epoch):

    agent.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets).cuda()
            if not args.parallel:
                inputs = inputs.cuda()

            probs, _ = agent(inputs)

            policy = probs.data.clone()
            policy[policy<0.5] = 0.0
            policy[policy>=0.5] = 1.0
            policy = Variable(policy)

            if args.cl_step < num_blocks:
                policy[:, :-args.cl_step] = 1

            preds = rnet(inputs, policy)
            reward, match = get_reward(preds, targets, policy.data)

            matches.append(match)
            rewards.append(reward)
            policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'TS - A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))
    print (log_str)

    log_value('test_accuracy', accuracy, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()

    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7'%(epoch, accuracy, reward, sparsity, len(policy_set)))


#--------------------------------------------------------------------------------------------------------#
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=10)


rnet = resneXt_cifar(56, 4, 16, num_classes=100, is_separate=True)
agent = Policy32([1,1,1], num_blocks=18)

checkpoint = torch.load('./checkpoint/resnext_56_4_16_config1_170.t7')#resnext_56_4_16_171.t7')
rnet.load_state_dict(checkpoint['net'])

# rnet, agent = utils.get_model(args.model)
num_blocks = 18

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print ('loaded agent from', args.load)

if args.parallel:
    agent = nn.DataParallel(agent)
    rnet = nn.DataParallel(rnet)

rnet.eval().cuda()
agent.cuda()

optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.wd)

configure(args.cv_dir+'/log', flush_secs=5)
lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)

    if args.cl_step < num_blocks:
        args.cl_step = 1 + 1 * (epoch // 1)
    else:
        args.cl_step = num_blocks

#     print ('training the last %d blocks ...' % args.cl_step)
    train(epoch)

    if epoch % 10 == 0:
        test(epoch)
