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
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoints')
parser.add_argument('--model', '-m', default='VGG16', help='type of model')
parser.add_argument('--name', '-n', default='ckpt', help='name for save file')
parser.add_argument('--optimizer', '-o', default='SGD-with-momentum', help='type of optimizer')
parser.add_argument('--epochs-to-run', '-e', default=200, type=int, help='num of epochs to run')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
acc_arr = []
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoints epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = 0
if args.model == "VGG16":
    net = VGG('VGG16')
elif args.model == "VGG19":
    net = VGG('VGG19')
elif args.model == "ResNet":
    net = ResNet18()
else:
    print("Wrong model: {}. exiting...".format(args.model))
    exit(1)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoints.
    print('==> Resuming from checkpoints..')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoints directory found!'
    checkpoint = torch.load('./checkpoints/{}.t7'.format(args.name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss() #Loss function

# Choose optimizer
if args.optimizer == "SGD":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=5e-4) # SGD Vanilla
elif args.optimizer == "SGD-with-momentum":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # SGD with momentum
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4) # Adam
else:
    print("Wrong optimizer, exiting...")
    exit(1)

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
        loss = criterion(outputs, targets) #Calculate loss function
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
    test_loss = 0
    correct = 0
    total = 0
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

    # Save checkpoints.
    acc = 100.*correct/total
    acc_arr.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_arr': acc_arr
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/{}.t7'.format(args.name))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + args.epochs_to_run):
    train(epoch)
    test(epoch)
