import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from densenet_cifar import *
from inference_2 import *
# from mobilenetv2 import *
# device = 'cuda'
# net = densenet_BC_cifar(depth=100, k=12, num_classes=100)
# net = net.to(device)

# checkpoint = torch.load('./dense100c100_190_21.91.t7')
# net.load_state_dict(checkpoint['net'], strict=False)
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']

# net = net.to('cpu')
# state = {
#         'net': net.state_dict(),
#         'acc': best_acc,
#         'epoch': start_epoch,
#     }

class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        # self.module = densenet_BC_cifar(depth=100, k=12, num_classes=100) # that I actually define.
        self.module = resnet110_cifar(num_classes=100)
    def forward(self, x):
        return self.module(x)

# then I load the weights I save from previous code:
# net = WrappedModel()
net = resnet110_cifar(num_classes=100)

checkpoint = torch.load('../checkpoint/resnet110c100_157.t7', map_location={"cuda" : "cpu"})
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': start_epoch,
    }

torch.save(state, 'resnet110c100_157_cpu.t7')