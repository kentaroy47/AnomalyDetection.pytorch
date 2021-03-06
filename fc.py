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

# import models
from models import fc_model, model_loss
from dataset import anomaly_dataset

### model options ###
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', default=30, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--train_file_name', '-train_name', type=str, default='./data/sin/normal.csv',
                    help='the file name of the training data set') 
parser.add_argument('--test_file_name', '-test_name', type=str, default='./data/sin/abn.csv',
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

Split_train_data_x = Split_train_data[0:-1,:].astype(np.float32)
Split_train_data_y = Split_train_data[1:,:].astype(np.float32)

Split_test_data_x = Split_test_data[0:-1,:].astype(np.float32)
Split_test_data_y = Split_test_data[1:,:].astype(np.float32)

img_rows = D

input_shape = (img_rows, 1, 1)

# define dataset    
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

# set model
model = fc_model([400, 200, 100, 50, 100, 200, 300, 400])
print(model)

# define loss
criterion = model_loss()

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

log_loss = []

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
    log_loss.append(epoch_train_loss)

import os
if not os.path.isdir("weights"): os.mkdir("weights")
torch.save(model.state_dict(), 'weights/fc' +
               str(epoch+1) + '.pth')

# evaluate
model.eval()
predict = model(torch.from_numpy(Split_test_data_x))

measured = Split_test_data_y.reshape(799*400)
predicted = predict.detach().numpy().reshape(799*400)

Loss_model=np.power(measured-predicted,2)
mean_window = 1000
Loss_model_processed = Loss_model[0:Loss_model.size-mean_window]

# smoothen anomaly score
for x in range(Loss_model.size-mean_window):
    Loss_model_processed[x] = np.mean(Loss_model[x:x+mean_window])
# normalize the score
Loss_model_processed = Loss_model_processed/(np.std(Loss_model_processed))
    
##### plot results #####
if not os.path.isdir("figs"): os.mkdir("figs")
fig0 = plt.figure()
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.plot(log_loss, label='trainloss')
plt.legend()
plt.show()
fig0.savefig('figs/FC_trainloss.png')

fig1 = plt.figure()
plt.xlabel("sample")
plt.ylabel("anomaly score")
plt.plot(Loss_model_processed, label='FC model score')
plt.legend()
plt.show()

fig1.savefig('figs/FC_anomaly_score.png')

fig2 = plt.figure()
plt.xlabel("sample")
plt.ylabel("value")
plt.plot(predicted[157500:163000], label='Pytorch FC model prediction')
plt.plot(measured[157500:163000], label='real data')
plt.legend()
plt.show()

fig2.savefig("figs/FC_waveforms.png")

fig3 = plt.figure()
plt.xlabel("sample")
plt.ylabel("value")
plt.plot(measured[0:3000], label='real data')
plt.plot(predicted[0:3000], label='Pytorch FC model prediction')
plt.legend()
plt.show()

fig3.savefig("figs/normal_waveform_predict.png")

