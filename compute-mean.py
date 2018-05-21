#!/usr/bin/env python3
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import numpy as np


def compute():
    trainTransform = transforms.Compose([
        transforms.Scale([256,256]),
        transforms.ToTensor(),
    ])
    
    trainLoader = DataLoader(
        dset.ImageFolder('/home/thisis928/hand_test',
                         transform=trainTransform
                         ),
        batch_size=16, shuffle=True)
        
    total_data = []
    for batch_idx, (data, target) in enumerate(trainLoader):
        if not len(total_data):
            total_data = data
            total_data = np.array(total_data)
        else:
            total_data = np.concatenate([data, total_data])
            
    means = []
    stdevs = []
    for i in range(3):
        pixels = total_data[:,i,:,:].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
        
    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
    
    return means, stdevs
    
if __name__=='__main__':
    compute()
