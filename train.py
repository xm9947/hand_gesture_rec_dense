#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

import densenet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--pretrain', type=str, default='work/densenet.base/latest.pth')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save) and args.pretrain == 'no':
        shutil.rmtree(args.save)
        print args.save
        os.makedirs(args.save, 0777)
    
    
    normMean = [0.50615275, 0.46781206, 0.44501185]
    normStd = [0.27191585, 0.26291886, 0.27136254]
    normTransform = transforms.Normalize(normMean, normStd)
    
    normMean_test = [0.5115661, 0.47129315, 0.44686756]
    normStd_test = [0.2853559, 0.27411368, 0.2873585]
    normTransform_test = transforms.Normalize(normMean_test, normStd_test)
    
    
    trainTransform = transforms.Compose([
        transforms.Scale([112,112]),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
       transforms.Scale([112,112]),
        transforms.ToTensor(),
        normTransform_test
    ])
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    '''
    trainLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=True, download=True,
                     transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=False, download=True,
                     transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)
    '''
    
    trainLoader = DataLoader(
        dset.ImageFolder('/home/thisis928/hand_data',
                         transform=trainTransform
                         ),
        batch_size=args.batchSz, shuffle=True, **kwargs)
        
    testLoader = DataLoader(
        dset.ImageFolder('/home/thisis928/hand_test',
                         transform=testTransform
                         ),
        batch_size=args.batchSz, shuffle=False, **kwargs)
        
    if args.pretrain == 'no':
        net = densenet.DenseNet(growthRate=40, depth=70, reduction=0.5,
                            bottleneck=True, nClasses=8)
    else:
        net = torch.load('work/densenet.base/latest.pth')
    

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #print data.shape
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
