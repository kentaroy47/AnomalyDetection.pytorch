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
parser.add_argument('--output_file_name', default='log')
parser.set_defaults(test=False)
                    
args = parser.parse_args()
n_epoch = args.epoch   # number of epochs

outputn=args.output_file_name
train_name=args.train_file_name
test_name=args.test_file_name
D=args.window_size #the size of the window width 
batchsize = args.batchsize   # minibatch size
epoch=args.epoch

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
#Split_test_data = Split_test_data.reshape(Split_test_data.shape[0], img_rows, 1 , 1)
#Split_train_data = Split_train_data.reshape(Split_train_data.shape[0], img_rows, 1 , 1)
#Split_test_data_y = Split_test_data_y.reshape(Split_test_data.shape[0], img_rows)
#Split_train_data_y = Split_train_data_y.reshape(Split_train_data.shape[0], img_rows)

input_shape = (img_rows, 1, 1)

class anomaly_dataset(torch.utils.data.Dataset):
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

batch_size = 4

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
print(images.size())  # torch.Size([4, 3, 300, 300])
print(len(targets))
print(targets[0].shape)  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数

