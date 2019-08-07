'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import math
import os, sys, random
import argparse
import socket
import threading
import time
import pickle
from multiprocessing import Process, Manager

from utils import progress_bar

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, baseWidth, stride=1, downsample=None, separate_coef=1):
        super(Bottleneck, self).__init__()
        D = int(planes * (baseWidth / (16*separate_coef)))##
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, groups=separate_coef, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
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
        
        out += residual
        out = self.relu(out)
        
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

    def forward(self, x=None, server_id=0):
        global feature_map
        l_send = 0
        l_rec = 0
        
        if x is None:
            while feature_map['receive'] is None:
                continue
            x = feature_map['receive']
            feature_map['receive'] = None
            skip = True
#             time.sleep(0.01)
        else:
            for i in [3,2,1]:
                while feature_map['send'] is not None:
                    continue
                feature_map['destination'] = i
                feature_map['send'] = x
            skip = False
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        device_num = self.cardinality
        s = 0
        ts = time.time()
        for stage in [self.layer1, self.layer2, self.layer3]:
            a = None
            for i, layer in enumerate(list(stage.modules())[0]):

                t1 = time.time()
#                 y = layer(x)
                x = layer(x)
                t2 = time.time()
                time2sleep = t2-t1
                time.sleep(time2sleep*0)#############
                
                if i%2==0:
                    time.sleep(0.005)
                    t1 = time.time()
                    while feature_map['send'] is not None:
                        continue
                    t2 = time.time()
                    l_send+=(t2-t1)
#                     time.sleep(0.001)
                    feature_map['destination'] = (server_id-s%3-1)%4
                    feature_map['send'] = x
                    s+=1
                    
                elif i%2==1:
                    t1 = time.time()
                    while feature_map['receive'] is None:
                        continue
                    t2 = time.time()
                    l_rec+=(t2-t1)
                    try:
                        x = (x+feature_map['receive'])/2
#                         print(x.shape)
                    except Exception as e:
                        print(x.shape)
                        print(feature_map['receive'].shape)
                    
                    feature_map['receive'] = None
                    if feature_map['receive_backup'] is not None:
                        feature_map['receive'] = feature_map['receive_backup']
                        feature_map['receive_backup'] = None
#                         print('get one')
                
#                 time.sleep(0.005)
        
        te = time.time()
#         print("%.3f, %.3f, %.3f"%(l_send, l_rec, te-ts))
        if skip:
            return
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resneXt_cifar(depth, cardinality, baseWidth, **kwargs):
    assert (depth - 2) % 9 == 0
    n = (depth - 2) // 9
    model = ResNeXt_Cifar(Bottleneck, [n, n, n], cardinality, baseWidth, **kwargs)
    return model



def receive_from_client(feature_map, server_id):
    
    print ("Starting server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if server_id == 0:
        s.bind(('127.0.0.1', 5000))
        device = 'cpu' 
    elif server_id == 1:
        s.bind(('127.0.0.1', 5001))
        device = 'cpu' 
    elif server_id == 2:
        s.bind(('127.0.0.1', 5002))
        device = 'cpu'
    elif server_id == 3:
        s.bind(('127.0.0.1', 5003))
        device = 'cpu' 
    s.listen(5)
    print("Server is up.")
    
#     global feature_map
    while True:
        
        (conn, address) = s.accept()
        data = conn.recv(1024)
        data_stream_size = pickle.loads(data)

        
        data_body = bytes()
        while sys.getsizeof(data_body)<data_stream_size:
            d = conn.recv(65535)
            if d:
                data_body += d
                
            else:
                break
        try:
            data = pickle.loads(data_body)
        except Exception as e:
            print("going to receiving %d"%data_stream_size)
            print(sys.getsizeof(data_body))
        
        if feature_map['receive'] is not None:
            while feature_map['receive_backup'] is not None:
                continue
            feature_map['receive_backup']=data[0]
        else:
            feature_map['receive']=data[0]
            
#         while feature_map['receive'] is not None:
#             continue
#         feature_map['receive']=data[0]
        
#         print("received %s @ %.4f"%(data[0], time.time()%1000))
#         print("internal_result received")

        
def send_to_server(feature_map, server_id):

#     print(feature_map)
#     print(feature_map['send'] is None)

    while True:
        if feature_map['send'] is not None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connected = False
            while not connected:
                try:
        #                 print("trying connecting to server...")
                    if feature_map['destination'] == 0:
                        s.connect(('127.0.0.1', 5000))
                    elif feature_map['destination'] == 1:
                        s.connect(('127.0.0.1', 5001))
                    elif feature_map['destination'] == 2:
                        s.connect(('127.0.0.1', 5002))
                    elif feature_map['destination'] == 3:
                        s.connect(('127.0.0.1', 5003))
                    connected = True
#                     print("Connected.")
                except Exception as e:
                    pass #Do nothing, just try again
#                     print("sent %s @ %.4f"%(feature, time.time()%1000))
            data_stream = pickle.dumps([feature_map['send']])
            data_stream_size = sys.getsizeof(data_stream)
            s.send(pickle.dumps(data_stream_size))
            time.sleep(0.001)
            s.sendall(data_stream)
#                     print(sys.getsizeof(data_stream))
#                     s.close()###############
#                     connected = False
#                     print("internal_result sent")
            feature_map['send'] = None
            s.close()
        time.sleep(0.001)


def server(server_id, self_ip=None, to_ip=None):
    global feature_map
    feature_map = Manager().dict({'send': None, 'receive': None, 'receive_backup': None, 'destination': None})
#     print(feature_map['send'])
#     print(type(feature_map))
    p1 = Process(target=receive_from_client, args=(feature_map, server_id))
    p1.start()
    
    p2 = Process(target=send_to_server, args=(feature_map, server_id))
    p2.start()
#     time.sleep(5)

    if server_id == 0:

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False) #, num_workers=1

        criterion = nn.CrossEntropyLoss()
        
#         net = resneXt_cifar(56, 4, 16, num_classes=100, is_separate=True)
#         print(net)
#         print(sum(p.numel() for p in net.parameters() if p.requires_grad))

        net = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        checkpoint = torch.load('./checkpoint/archived/resnext_56_4_16_config1_170.t7')

        for key in checkpoint['net'].keys():
            if 'layer' in key and 'num_batches_tracked' not in key:
                try:
                    checkpoint['net'][key] = torch.split(checkpoint['net'][key], checkpoint['net'][key].size(0)//4, 0)[0]
                except Exception as e:
                    print(checkpoint['net'][key].shape)
        
        net.load_state_dict(checkpoint['net'], strict=False)

        print(sum(p.numel() for p in net.parameters() if p.requires_grad))


        def test(epoch=0):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            
                    
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
    #                 inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
        #             outputs = (net(inputs)+net2(inputs)+net3(inputs)+net5(inputs))
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#                     time.sleep(1)


        test()

    else:
#         net = resneXt_cifar(56, 4, 16, num_classes=100, is_separate=True)
#         print(net)
        net = resneXt_cifar(56, 1, 16, num_classes=100, is_separate=True)
#         print(net)
        net.eval()
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        checkpoint = torch.load('./checkpoint/archived/resnext_56_4_16_config1_170.t7')

        for key in checkpoint['net'].keys():
#             print(key)
            if 'layer' in key and 'num_batches_tracked' not in key:
                try:
                    
#                     print(checkpoint['net'][key].shape)
                    checkpoint['net'][key] = torch.split(checkpoint['net'][key], checkpoint['net'][key].size(0)//4, 0)[server_id]
#                     print(checkpoint['net'][key].shape)
                except Exception as e:
                    print(checkpoint['net'][key].shape)
#             if 'fc' in key:
                
        
        net.load_state_dict(checkpoint['net'], strict=True)

        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('done')
        with torch.no_grad():
            while True:
                if feature_map['receive'] is not None:
                    net.forward(server_id=server_id)



def parse_arguments():
    parser = argparse.ArgumentParser()
#     parser.add_argument('-c', dest='client', action='store_true')
    parser.add_argument('--id', dest='server_id', type=int, default=0)
#     parser.add_argument('--sip', dest='self_ip', type=str)
#     parser.add_argument('--tip', dest='to_ip', type=str)
    result = parser.parse_args()
    return result

def main():
    args = parse_arguments()
    try:
        server(args.server_id)
    except KeyboardInterrupt:
        print ("Keyboard interrupt")

if __name__ == '__main__':
    main()

    

#python3 main.py --lr 0.1 --split 1 --epoch 200 --batch 128 --schedule 50 --cuda 1 --save resnext_56_2_16