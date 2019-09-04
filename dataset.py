# -*- coding: utf-8 -*-

import torch.utils.data as data

# define dataset
class anomaly_dataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]