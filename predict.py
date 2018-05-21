# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader



def predict():
    normMean = [0.50615275, 0.46781206, 0.44501185]
    normStd = [0.27191585, 0.26291886, 0.27136254]
    normTransform = transforms.Normalize(normMean, normStd)
    
    testTransform = transforms.Compose([
       transforms.Scale([112,112]),
        transforms.ToTensor(),
        normTransform
    ])
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    testLoader = DataLoader(
        dset.ImageFolder('data',
                         transform=testTransform
                         ), 
                         batch_size=32,shuffle=False, **kwargs)
    
    model = torch.load('work/densenet.base/latest.pth')
    model.eval()
    
    pred = -1
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        
    f = open('tmp.txt', "wb")
    f.write(str(pred[0]))
    f.close()

if __name__=='__main__':
    predict()