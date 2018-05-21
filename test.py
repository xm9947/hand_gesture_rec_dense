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


def test_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    normMean = [0.49505615, 0.46180186, 0.4387954]
    normStd = [0.24403621, 0.2434699, 0.24818134]
    normTransform = transforms.Normalize(normMean, normStd)
    
    testTransform = transforms.Compose([
       transforms.Scale([112,112]),
        transforms.ToTensor(),
        normTransform
    ])
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    testLoader = DataLoader(
        dset.ImageFolder('/home/thisis928/hand_val',
                         transform=testTransform
                         ), 
                         batch_size=32,shuffle=False, **kwargs)
    
    model = torch.load('work/densenet.base/latest.pth')
    model.eval()
    test_loss = 0
    incorrect = 0
    
    drop_len = 0
    for data, target in testLoader:
        if data.shape[0] == 1:
            drop_len = 1
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()
        print target.data

    test_loss = test_loss
    test_loss /= len(testLoader) - drop_len # loss function already averages over batch size
    nTotal = len(testLoader.dataset) - drop_len
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))
    
    
if __name__=='__main__':
    test_data()