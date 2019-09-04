# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

# define model here
class fc_model(nn.Module):
    def __init__(self, arch):
        super(fc_model, self).__init__()
        self.arch = arch
        self.fc1 = nn.Linear(arch[0], arch[1])
        self.bn1 = nn.BatchNorm1d(arch[1])
        self.fc2 = nn.Linear(arch[1], arch[2])
        self.bn2 = nn.BatchNorm1d(arch[2])
        self.fc3 = nn.Linear(arch[2], arch[3])
        self.bn3 = nn.BatchNorm1d(arch[3])
        self.fc4 = nn.Linear(arch[3], arch[4])
        self.bn4 = nn.BatchNorm1d(arch[4])
        self.fc5 = nn.Linear(arch[4], arch[5])
        self.bn5 = nn.BatchNorm1d(arch[5])
        self.fc6 = nn.Linear(arch[5], arch[6])
        self.bn6 = nn.BatchNorm1d(arch[6])
        self.fc7 = nn.Linear(arch[6], arch[7])
        
    def forward(self, x):
        # network performed well without batchnorm
        x = F.leaky_relu(self.fc1(x), inplace=True)
        x = F.leaky_relu(self.fc2(x), inplace=True)
        x = F.leaky_relu(self.fc3(x), inplace=True)
        x = F.leaky_relu(self.fc4(x), inplace=True)
        x = F.leaky_relu(self.fc5(x), inplace=True)
        x = F.leaky_relu(self.fc6(x), inplace=True)
        x = self.fc7(x)

        return x
    
# define loss. use MSE here
class model_loss(nn.Module):
    def __init__(self):
        super(model_loss, self).__init__()
    
    def forward(self, x, target):
        return F.mse_loss(x, target)