# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:45:08 2019

@author: my_lab
"""
#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from utils_ import *
#%% Set Hyper Parameter 超参数设置
class Parameter:
    # 超参数类
    def __init__(self):
        # 数据参数
        self.datanum = 6877# 提取数据量
        self.trainratio = 0.9  # 训练集的比例
        # 训练参数
        self.batch_size = 50 # Batch的大小
        self.epoch =  50 # 迭代次数
        self.lr = 0.001  # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
        self.ifGPU = False  # 是否需要使用GPU（需要电脑里有NVIDIA的独立显卡并安装CUDA驱动才可以)


PARM = Parameter()
PARM.ifGPU = True  # 开启GPU
PATH = 'E:\\zhumin\\TrainingSet1\\TrainingSet1\\'
LABEL = 'REFERENCE.csv'
# %% Input Data 数据输入
 # 特征数据
count = 0
ECG_feature_data = []
ages = []
for file in os.listdir(PATH):
    count += 1
    if count > PARM.datanum:
        break

    if file.endswith('.mat'):
        load_data = scio.loadmat(PATH + file)  # 读取mat格式数据
        sample = dict()  # 数据以字典形式存储
        sample['sex'] = str(load_data['ECG']['sex'][0][0][0])  # 性别
        try:
            sample['age'] = int(load_data['ECG']['age'][0][0][0][0])  # 年龄
        except:
            sample['age'] = 60
        ages.append(sample['age'])
        sample['data'] = load_data['ECG']['data'][0][0] 
        
        # ECG数据
        sample['data'] = sample['data'][:,:int(sample['data'].shape[1]//4*4)]
        sample['data'] = sample['data'].reshape(sample['data'].shape[0],-1,4)
        sample['data'] = np.average(sample['data'],axis=2)
        sample['data'] = remove_outlier(sample['data'])
        sample['data'] = remove_noise(sample['data'])

        ECG_feature_data.append(sample)  # 存入特征数据中
# 标签数据
ECG_label_data = pd.read_csv(PATH + LABEL)
ECG_label_data = ECG_label_data.iloc[0: PARM.datanum]  # 存入标签数据
# %% Reprocess Data 数据预处理
# 性别转换成特征
sex_change = {'Male': [1, 0], 'Female': [0, 1]}
# 年龄标准化
ages = np.array(ages)

def z_score(age, mean=ages.mean(), std=ages.std()):
    return (age - mean) / std

# 训练数据
X = []
for i in ECG_feature_data:
    sample = dict()
    sample['data'] = torch.FloatTensor(tosamples0(i['data']))
    sample['sex'] = sex_change[i['sex']]
    sample['age'] = z_score(i['age'])
    X.append(sample)
# 测试数据
Y = np.array(list(ECG_label_data['First_label']))
Y -= 1  # 从0开始
# 训练集测试集分割
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    test_size=1 - PARM.trainratio,
                                                    random_state=0,
                                                    stratify=Y)
# 标签变成LongTensor格式
train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
test_y = torch.FloatTensor(test_y).type(torch.LongTensor)
# 删除不用的数据(防止占内存)
del X, Y, ECG_feature_data, ECG_label_data, ages, file, load_data, sample, sex_change
# %% Load Data 装载数据
def dataloader(feature, label, batch_size=1, shuffle=True):
    L = len(feature)
    dataset = []
    for i in range(L):
        dataset.append([feature[i], label[i]])
    if shuffle:
        np.random.shuffle(dataset)
    batch = []
    for i in range(0, L, batch_size):
        f = [j[0] for j in dataset[i: i + batch_size]]
        l = [j[1] for j in dataset[i: i + batch_size]]
        batch.append([f, l])
    return batch


# load data
loader = dataloader(train_x, train_y, PARM.batch_size, True)
test_loader = dataloader(test_x, test_y, 100, True)
#%% Create Network 网络构建
class CNN_Linear_RNN4(nn.Module):
    def __init__(self):
        super(CNN_Linear_RNN4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, (5, 1), 1, 0 ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.bilstm1 =nn.LSTM(256,256,bidirectional=True)
        self.bilstm2 =nn.LSTM(512,512,bidirectional=True)
        self.network3 = nn.Sequential(
                nn.Linear(1024, 512),
                # nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(512, 100),
                nn.Dropout(0.5),
                nn.Linear(100,4)
                )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], 256, -1) # win_num * 256 * 51
        x = x.permute(2, 0, 1)             # 31 * win_num * 256
        r_out, _ = self.bilstm1(x)         # 31 * win_num * 512
        _, (h_n, _) = self.bilstm2(r_out)  # 2 * win_num * 512
#        print(h_n.shape)
        x = h_n.permute(1, 0, 2)           # win_num * 2 * 512
        x = x.reshape(x.shape[0], -1)      # win_num * 1024
        x = self.network3(x)
        return x
#%%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(16,64)
        self.networks = nn.Sequential(
                nn.Linear(64,64),
                nn.Dropout(0.5),
                nn.Linear(64,9)
                )
    def forward(self,x):
        x = x.permute(1,0,2) 
        x,(h_n,c_n)= self.lstm(x)
        x = h_n[0, :, :]
#        print(h_n.shape)
        x = self.networks(x)
        return x
#%%
cnn_rnn1 = CNN_Linear_RNN4()
cnn_rnn2 = CNN_Linear_RNN4()
cnn_rnn3 = CNN_Linear_RNN4()
cnn_rnn4 = CNN_Linear_RNN4()

lstm =LSTM()
if PARM.ifGPU:
    cnn_rnn1 = cnn_rnn1.cuda()
    cnn_rnn2 = cnn_rnn2.cuda()
    cnn_rnn3 = cnn_rnn3.cuda()
    cnn_rnn4 = cnn_rnn4.cuda()
    lstm =LSTM().cuda()
#%%
optimizer = torch.optim.Adam([{'params': cnn_rnn1.parameters()},
                              {'params': cnn_rnn2.parameters()},
                              {'params': cnn_rnn3.parameters()},
                              {'params': cnn_rnn4.parameters()},
                              {'params': lstm.parameters()}], lr=PARM.lr)
loss_func = nn.CrossEntropyLoss()
    
#%%
for epoch in range(PARM.epoch):
    for step, (x, y) in enumerate(loader):
        output = []
        for i in x: 
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            result1 = cnn_rnn1(input)
            result2 = cnn_rnn2(input)
            result3 = cnn_rnn3(input)
            result4 = cnn_rnn4(input)
            results = torch.cat((result1,result2,result3,result4),dim=1)
            l = len(results)
            if l<45:
                    z = torch.zeros((45-l,16)).cuda()
                    results = torch.cat((z,results),dim=0)
            else:
                results = results[:45,:]
            result = lstm(results[np.newaxis,:,:])
            output.append(result)
        output = torch.cat(output)
        y = torch.FloatTensor(y).type(torch.LongTensor)
        if PARM.ifGPU:
            y = y.cuda()
        loss = loss_func(output, y)  # loss
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if step % 1 == 0:
            if PARM.ifGPU:
                pred = torch.max(output, 1)[1].cuda().data.squeeze()
            else:
                pred = torch.max(output, 1)[1].data.squeeze()
            accuracy = float((pred == y).sum()) / float(y.size(0))
            # F1 = score_s(pred, y)
            print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f '
                  % (epoch, step, loss.data, accuracy))
            del x, y, input, output
            if PARM.ifGPU: torch.cuda.empty_cache()
    num = 0
    correct_num = 0   
    all_y = []
    all_pred = []
    for step, (x, y) in enumerate(test_loader): 
        cnn_rnn1.eval()
        cnn_rnn2.eval()
        cnn_rnn3.eval()
        cnn_rnn4.eval()# test model
        lstm.eval()
        output = []
        for i in x:
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            result1 = cnn_rnn1(input)
            result2 = cnn_rnn2(input)
            result3 = cnn_rnn3(input)
            result4 = cnn_rnn4(input)
            results = torch.cat((result1,result2,result3,result4),dim=1)
            l = len(results)
            if l<45:
                    z = torch.zeros((45-l,16)).cuda()
                    results = torch.cat((z,results),dim=0)
            else:
                results = results[:45,:]
            result = lstm(results[np.newaxis,:,:])
            output.append(result)
        output = torch.cat(output)
        y = torch.FloatTensor(y).type(torch.LongTensor)
        if PARM.ifGPU:
            y = y.cuda()
        if PARM.ifGPU:
            pred = torch.max(output, 1)[1].cuda().data.squeeze()
        else:
            pred = torch.max(output, 1)[1].data.squeeze()
        all_y.append(y)
        all_pred.append(pred)
    # evaluate
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    accuracy = float((pred== y).sum()) / float(y.size(0))
    print('Epoch: %s | test accuracy: %.2f'% (epoch, accuracy))
    del x, y, all_pred, all_y, input, output
    if PARM.ifGPU: torch.cuda.empty_cache() # empty GPU memory