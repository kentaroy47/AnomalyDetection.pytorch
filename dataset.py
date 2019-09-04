# -*- coding: utf-8 -*-

import torch.utils.data as data
import numpy as np

# define dataset
class anomaly_dataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def pullitem(self, index):
        # pull and normalize
        img = self.data[index]
        img -= np.mean(img)
        img /= np.std(img)
        
        target = self.target[index]
        target -= np.mean(target)
        target /= np.std(target)
        
        return img, target
        
    def __getitem__(self, index):
        img, target = self.pullitem(index)
        return img, target
    
class classify_anomaly_dataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def pullitem(self, index):
        # pull and normalize
        img = self.data[index]
        img -= np.mean(img)
        img /= np.std(img)
        
        return img
        
    def __getitem__(self, index):
        img = self.pullitem(index)
        return img, self.target[index]
    
    