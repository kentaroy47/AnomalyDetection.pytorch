# -*- coding: utf-8 -*-
import argparse
import math
import sys
import time
import copy
import matplotlib.pylab as plt
#matplotlib.use('Agg') # for AWS

import numpy as np
from numpy .random import multivariate_normal, permutation

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### model options ###
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', default=30, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--train_file_name', '-train_name', type=str, default='./data/normal.csv',
                    help='the file name of the training data set') 
parser.add_argument('--test_file_name', '-test_name', type=str, default='./data/abn.csv',
                    help='the file name of the test data set')
parser.add_argument('--window_size', '-ws', type=int, default=400,
                    help='window size')
parser.add_argument('--lr', '-l', type=float, default=1e-2,
                    help='learn rate')
parser.add_argument('--output_file_name', default='log')
parser.set_defaults(test=False)
                    
args = parser.parse_args()

# set parser to var
outputn = args.output_file_name
train_name = args.train_file_name
test_name = args.test_file_name
D = args.window_size #the size of the window width 
batch_size = args.batchsize   # minibatch size
num_epoch=args.epoch

###### data preparation #####

# load
x_train_data=np.loadtxt(train_name,delimiter=',')
x_test_data=np.loadtxt(test_name,delimiter=',')

# normalization
x_train_data = x_train_data/(np.std(x_train_data))
x_test_data = x_test_data/(np.std(x_test_data))

print("x_train data", x_train_data.shape)
print("x_test data", x_test_data.shape)

"""
split train data and test data into D length sequences.
the keras model will try to predict the *next* D length sequence.
if the model results and the real data has no contradictions, the state is non-anomaly (or nomal)
if the model results and the real data have large differences, it is likely to be an anomaly state.
# 2019/07/04 fixed to remove divide errors.
"""
x_train_data = x_train_data.reshape([x_train_data.shape[0], 1]) #x_train_data.size/x_train_data.shape[0]])
x_test_data = x_test_data.reshape([x_test_data.shape[0], 1]) #x_test_data.size/x_test_data.shape[0]])

print("x_train data", x_train_data.shape)
print("x_test data", x_test_data.shape)

Split_train_data = x_train_data.reshape([int(x_train_data.shape[0]/D), x_train_data.shape[1]*D])
Split_test_data=x_test_data.reshape([int(x_test_data.shape[0]/D), x_test_data.shape[1]*D])

Split_train_data_x=Split_train_data[0:-1,:]
Split_train_data_y=Split_train_data[1:,:]

Split_test_data_x=Split_test_data[0:-1,:]
Split_test_data_y=Split_test_data[1:,:]

img_rows = D

input_shape = (img_rows, 1, 1)

# define dataset
class anomaly_dataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
train_dataset = anomaly_dataset(Split_train_data_x, Split_train_data_y)
val_dataset = anomaly_dataset(Split_test_data_x, Split_test_data_y)

# データローダーの作成
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
print(images.size())
print("batch len is ", len(targets))
print(targets[0].shape)  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数

# define model here
class fc_model(nn.Module):
    def __init__(self, arch):
        super(fc_model, self).__init__()
        
        self.fc1 = nn.Linear(arch[0], arch[1])
        self.fc2 = nn.Linear(arch[1], arch[2])
        self.fc3 = nn.Linear(arch[2], arch[3])
        self.fc4 = nn.Linear(arch[3], arch[4])
        self.fc5 = nn.Linear(arch[4], arch[5])
        self.fc6 = nn.Linear(arch[5], arch[6])
        self.fc7 = nn.Linear(arch[6], arch[7])
        
    def forward(self, x):
        x = nn.LeakyReLU(self.fc1(x))
        x = nn.LeakyReLU(self.fc2(x))
        x = nn.LeakyReLU(self.fc3(x))
        x = nn.LeakyReLU(self.fc4(x))
        x = nn.LeakyReLU(self.fc5(x))
        x = nn.LeakyReLU(self.fc6(x))
        x = self.fc7(x)

        return x

model = fc_model([400, 200, 100, 50, 100, 200, 300, 400])
print(model)

# define loss. use MSE here
class model_loss(nn.Module):
    def __init__(self):
        super(model_loss, self).__init__()
    
    def forward(self, x, target):
        return F.mse_loss(x, target)

criterion = model_loss

# define optimizer. use SGD here.
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9,
                      weight_decay=0.0001)



######### start training ##########
# enable GPUs if any.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

model.to(device)
model.train()

iteration = 1
phase = "train"

for epoch in range(num_epoch):
    # 開始時刻を保存
    t_epoch_start = time.time()
    t_iter_start = time.time()
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和

    print('-------------')
    print('Epoch {}/{}'.format(epoch+1, num_epoch))
    print('-------------')
    
    
    
    for imges, targets in dataloaders_dict["train"]:
        imges = imges.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(imges)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                t_iter_finish = time.time()
                duration = t_iter_finish - t_iter_start
                print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                    iteration, loss.item()/batch_size, duration))
                t_iter_start = time.time()

            epoch_train_loss += loss.item()
            iteration += 1
            
    # epochのphaseごとのlossと正解率
    t_epoch_finish = time.time()
    print('-------------')
    print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
        epoch+1, epoch_train_loss, 0))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
    t_epoch_start = time.time()

import os
if not os.path.isdir("weights"): os.mkdir("weights")
torch.save(model.state_dict(), 'weights/fc' +
               str(epoch+1) + '.pth')

