import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from utils import progress_bar
import argparse
import socket
import threading
import time
import pickle
import sys
from multiprocessing import Process, Manager

# global feature_map
# feature_map = {'x1':None, 'x2':None, 'y1':None, 'y2':None}

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
        
#         ### case 2 ###
#         self.layer1a = self._make_layer(block, 16, blocks=12, stride=1)
#         self.layer2a = self._make_layer(block, 32, blocks=6, stride=2)
#         self.layer3a = self._make_layer(block, 64, blocks=12, stride=2)
        
#         self.inplanes = 16
#         self.layer1b = self._make_layer(block, 16, blocks=6, stride=1)
#         self.layer2b = self._make_layer(block, 32, blocks=12, stride=2)
#         self.layer3b = self._make_layer(block, 64, blocks=6, stride=2)
#         ### end ###
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes) # group

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
        
#         y = self.layer1b(x)
#         x = self.layer1a(x)
#         x = x+y
        
#         x = self.layer2a(x)
#         y = self.layer2b(y)
#         y = x+y

#         y = self.layer3b(y)
#         x = self.layer3a(x)
#         x = x+y    

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x    
    
    def forward_d1(self, x):
        global feature_map
        t = time.time() ###
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        t1 = time.time() ###
        feature_map['x1'] = x1.detach()
        t2 = time.time() ###

        x = self.layer1a(x1)
        t3 = time.time() ###
        print_flag = True
        while True:
            if feature_map['y1'] is not None:
                x = x+feature_map['y1']
                feature_map['y1'] = None
                break
            if print_flag:
                print("no y1")
                print_flag = False
            time.sleep(0.005)
        t4 = time.time() ###
        x2 = self.layer2a(x)
        t5 = time.time() ###
        feature_map['x2'] = x2.detach()
        t6 = time.time() ###

        x = self.layer3a(x2)
        t7 = time.time() ###
        print_flag = True
        while True:
            if feature_map['y2'] is not None:
                t8 = time.time() ###
                x = x+feature_map['y2']
                feature_map['y2'] = None
                break
            if print_flag:
                print("no y2")
                print_flag = False
            time.sleep(0.005)
        t9 = time.time() ###
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        t10 = time.time() ###
        print("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n"%(t1-t, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t9-t8, t10-t9))
        return x  
    
    def forward_d2(self, x1):
        global feature_map
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x1 = self.relu(x)
        
        
        y1 = self.layer1b(x1)
        feature_map['y1'] = y1.detach()
#         x = self.layer1a(x1)
#         x = x+y1
        
#         x2 = self.layer2a(x)
        y = self.layer2b(y1)
        print_flag = True
        while True:
            if feature_map['x2'] is not None:
                y = feature_map['x2']+y
                feature_map['x2'] = None
                break
            if print_flag:
                print("no x2")
                print_flag = False
            time.sleep(0.005)

        y2 = self.layer3b(y)
        feature_map['y2'] = y2.detach()
#         x = self.layer3a(x2)
#         x = x+y2    

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x 
        

def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def receive_from_client(conn, address, device, feature_map):
    
#     global feature_map
    while True:
        data = conn.recv(1024)
        data_stream_size = pickle.loads(data)
        print("going to receiving %d"%data_stream_size)
        
        data_body = bytes() 
        while sys.getsizeof(data_body)<data_stream_size:
            d = conn.recv(1024)
            if d:
                data_body += d
#                 print(sys.getsizeof(data_body))
            else:
                break
        data = pickle.loads(data_body)
        feature_map[data[0]]=data[1]
        print("received %s @ %.4f"%(data[0], time.time()%1000))
#         print("internal_result received")

        
def send_to_server(server_id, feature_map):
#     global feature_map
    print(feature_map)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if server_id == 1:
        connected = False
        while not connected:
            try:
#                 print("trying connecting to server...")
                s.connect(('192.168.11.40', 5000))
                connected = True
                print("Connected.")
            except Exception as e:
                pass #Do nothing, just try again
        while True:
            for feature in ['y1', 'y2']:
                if feature_map[feature] is not None:
                    print("sent %s @ %.4f"%(feature, time.time()%1000))
                    data_stream = pickle.dumps([feature, feature_map[feature]])
                    data_stream_size = sys.getsizeof(data_stream)
                    s.send(pickle.dumps(data_stream_size))
                    time.sleep(0.001)
                    s.sendall(data_stream)
#                     print(sys.getsizeof(data_stream))
#                     s.close()###############
#                     connected = False
#                     print("internal_result sent")
                    feature_map[feature] = None
                    
                time.sleep(0.005)
    elif server_id == 0:
        connected = False
        while not connected:
            try:
#                 print("trying connecting to server...")
                s.connect(('192.168.11.19', 5000))
                connected = True
                print("Connected.")
            except Exception as e:
                pass #Do nothing, just try again
        while True:
            for feature in ['x1', 'x2']:
                if feature_map[feature] is not None:
                    print("sent %s @ %.4f"%(feature, time.time()%1000))
                    data_stream = pickle.dumps([feature, feature_map[feature]])
                    data_stream_size = sys.getsizeof(data_stream)
                    s.send(pickle.dumps(data_stream_size))
                    time.sleep(0.001)
                    s.sendall(data_stream)
#                     print(sys.getsizeof(data_stream))
#                     s.close()##################
#                     connected = False
#                     print("internal_result sent")
                    feature_map[feature] = None
                    
                time.sleep(0.005)

def server(server_id):
    global feature_map
    feature_map = Manager().dict({'x1':None, 'x2':None, 'y1':None, 'y2':None})
    print ("Starting server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if server_id == 0:
        s.bind(('192.168.11.40', 5000))
        device = 'cpu' 
    elif server_id == 1:
        s.bind(('192.168.11.19', 5000))
        device = 'cpu' 
    s.listen(5)
    print("Server is up.")
    
#     t2 = threading.Thread(target=send_to_server, args=(server_id,))
#     t2.daemon = True
#     t2.start()
    p2 = Process(target=send_to_server, args=(server_id, feature_map))
    p2.start()
    
    (conn, address) = s.accept()
    
#     t1 = threading.Thread(target=receive_from_client, args=(conn, address, device))
#     t1.daemon = True
#     t1.start()
    p1 = Process(target=receive_from_client, args=(conn, address, device, feature_map))
    p1.start()

    
    if server_id == 0:
        
        # device = 'cpu' 


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False) #, num_workers=1

        criterion = nn.CrossEntropyLoss()


        net = resnet110_cifar(num_classes=100)
        net = net.to(device) ###

        checkpoint = torch.load('./checkpoint/resnet110_12_6c100_113_cpu.t7')
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
#                     inputs, targets = inputs.to(device), targets.to(device) ###
                    outputs = net.forward_d1(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


        test()

    elif server_id == 1:

        net = resnet110_cifar(num_classes=100)
#         net = net.to(device) ###
        net.eval()

        checkpoint = torch.load('./checkpoint/resnet110_12_6c100_113_cpu.t7')
        net.load_state_dict(checkpoint['net'], strict=False)

        while True:
            if feature_map['x1'] is not None:
                net.forward_d2(feature_map['x1'])
                feature_map['x1'] = None



def parse_arguments():
    parser = argparse.ArgumentParser()
#     parser.add_argument('-c', dest='client', action='store_true')
    parser.add_argument('--id', dest='server_id', type=int, default=0)
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
















# net = net.to('cpu')
# state = {
#     'net': net.state_dict(),
# }
# torch.save(state, './checkpoint/resnet164c100_118_rpi.t7')
