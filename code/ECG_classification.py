# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:24:05 2019

@author: QuYue
"""
#%% Import Packages
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from score_py3 import score

#%% Set Hyper Parameter 超参数设置
class Parameter:
    # 超参数类
    def __init__(self):
        # 数据参数
        self.datanum = 2000      # 提取数据量
        self.trainratio = 0.9   # 训练集的比例
        # 窗口参数
        self.winwidth = 4000 # 窗口宽度
        self.winstep = 2000  # 窗口移动步长
        # 训练参数
        self.batch_size =100    # Batch的大小
        self.epoch = 20          # 迭代次数
        self.lr = 0.001          # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
        self.ifGPU = False       # 是否需要使用GPU（需要电脑里有NVIDIA的独立显卡并安装CUDA驱动才可以)

PARM = Parameter()
PARM.ifGPU =  True      # 开启GPU
PATH = 'E:\\zhumin\\TrainingSet1\\TrainingSet1\\'
LABEL = 'REFERENCE.csv'

#%% Input Data 数据输入
# 特征数据
count = 0
ECG_feature_data = []
ages = []
for file in os.listdir(PATH):
    count +=1
    if count > PARM.datanum:
        break

    if file.endswith('.mat'):
        load_data = scio.loadmat(PATH+file) # 读取mat格式数据
        sample = dict()  #数据以字典形式存储
        sample['sex'] = str(load_data['ECG']['sex'][0][0][0])    #性别
        try:
            sample['age'] = int(load_data['ECG']['age'][0][0][0][0]) #年龄
        except:
            sample['age'] = 60
        ages.append(sample['age'])
        sample['data'] = load_data['ECG']['data'][0][0]          #ECG数据
        ECG_feature_data.append(sample) # 存入特征数据中
# 标签数据
ECG_label_data = pd.read_csv(PATH+LABEL)
ECG_label_data = ECG_label_data.iloc[0: PARM.datanum] # 存入标签数据

#%% Reprocess Data 数据预处理
# 窗口切割
def get_windows(data, winwidth, winstep): 
    # 窗口切割函数
    L = data.shape[1] # 数据的长度
    windows = []
    for i in range(0, L-winwidth+1, winstep):
        w = data[:, i: i+winwidth]
        w = w[np.newaxis, :, :, np.newaxis]
        windows.append(w)
    return windows
# 性别转换成特征
sex_change = {'Male': [1, 0], 'Female': [0, 1]}
# 年龄标准化
ages = np.array(ages)
def z_score(age, mean=ages.mean(), std=ages.std()):
    return (age - mean)/std

# 训练数据   
X = []
for i in ECG_feature_data:
    sample = dict()
    sample['data'] = get_windows(torch.FloatTensor(i['data']), PARM.winwidth, PARM.winstep)
    sample['sex'] = sex_change[i['sex']]
    sample['age'] = z_score(i['age'])    
    X.append(sample)
# 测试数据
Y = np.array(list(ECG_label_data['First_label']))
Y -= 1 # 从0开始
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

#%% Load Data 装载数据 
def dataloader(feature, label,  batch_size=1, shuffle=True):
    L = len(feature)
    dataset = []
    for i in range(L):
        dataset.append([feature[i], label[i]])
    if shuffle:
        np.random.shuffle(dataset)
    batch = []
    for i in range(0, L, batch_size):
        f = [j[0] for j in dataset[i: i+batch_size]]
        l = [j[1] for j in dataset[i: i+batch_size]]
        l = torch.FloatTensor(l).type(torch.LongTensor)
        batch.append([f, l])
    return batch

# load data
loader = dataloader(train_x, train_y, PARM.batch_size, True)
#%% Create Network 网络构建        
class CNN_Linear_RNN(nn.Module):
    def __init__(self):
        super(CNN_Linear_RNN, self).__init__()       
        # 第一个卷积层（包括了卷积、整流、池化）
        self.conv1 = nn.Sequential( 
            # 卷积 
            nn.Conv2d(
                in_channels=12,      # 输入通道
                out_channels=32,     # 输出通道
                kernel_size=(50, 1), # 卷积核的长度
                stride=1,           # 步长 （卷积核移动的步长）
                padding=0,          # 补0 zero_padding （对图片的边缘加上一圈0，防止越卷积图片越小）
            ),
            # Batch_normalization
            nn.BatchNorm2d(32),
            # 整流 使用ReLU函数进行整流
            nn.ReLU(),   
            # 池化 使用最大池化
            nn.MaxPool2d(kernel_size=(3,1)) # 最大池化 在3*1的范围内取最大
        )
        # 第二个卷积层（包括了卷积、整流、池化）
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (30, 1), 1, 0), # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0), # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3,1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0), # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3,1))
        )
        # batch*400*32
        self.network = nn.Sequential(
            nn.Linear(5760, 5000),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(5000,3000)
    
    def forward(self, x, last):
        # 输入（Batch_size * 12 * 3000 * 1）
        x = self.conv1(x) #（Batch_size *  1330 *  * 1）
        x = self.conv2(x) #（Batch_size * 32 * 440 * 1）
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.network(x) #（Batch_size * 128 * 1）

        r_out, h_n = self.lstm(x, last)
        return r_out, h_n


class Decisioner(nn.Module):
    def __init__(self):
        super(Decisioner, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(3003, 2000), # lstm output后加一个普通的神经网络
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(2000, 1000),  # lstm output后加一个普通的神经网络
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1000, 500),  # lstm output后加一个普通的神经网络
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(500, 9)
                )
        
    def forward(self, feature, new_feature):
        feature = torch.cat([feature, new_feature],1)
        output = self.network(feature)
        return output

class ECG_Class():
    def __init__(self, net1, net2, ifGPU):   
        self.cnn2rnn = net1
        self.decisioner = net2  
        self.ifGPU = ifGPU
        self.loss = None
        #  优化算法 使用adam算法来进行优化，lr是学习率，学习率越高，学习速度越快，越低，精度可能会越高
        self.optimizer = torch.optim.Adam([{'params': self.cnn2rnn.parameters()},
                              {'params': self.decisioner.parameters()}], lr=PARM.lr)
        # 损失函数    
        self.loss_func = nn.CrossEntropyLoss() # 损失函数（对于多分类问题使用交叉熵）
        # 
        if self.ifGPU:
            self.cnn2rnn = self.cnn2rnn.cuda()
            self.decisioner = self.decisioner.cuda()
        
        
    def get_new_feature(self, sex, age):
        # 新的特征
        new_feature = [sex[0], sex[1], age]
        new_feature = torch.FloatTensor([new_feature])
        if self.ifGPU:
            new_feature = new_feature.cuda()
        return new_feature
    
    def loss_sum(self, loss):
        # 一个batch的loss和并
        Loss = loss[0]
        for i in loss[1:]:
            Loss += i
        Loss /= len(loss)
        return Loss  
    
    def forward(self, x, y, if_loss= True):
        if if_loss:
            loss = []
        batch_output = []
        for i in range(len(x)): # 将batch的样本拆开 
            data = x[i]['data']      # 数据
            sex = x[i]['sex']        # 性别
            age = x[i]['age']        # 年龄

            label = y[i][np.newaxis] # 标签
            last = None              # rnn的上一层记忆
            new_feature = self.get_new_feature(sex, age)
            for windows in data:
                if self.ifGPU:
                    windows = windows.cuda()
                r_out, h_n = self.cnn2rnn(windows, last)
                last = [r_out, h_n]
            output = self.decisioner(r_out, new_feature)
            batch_output.append(output)
            if if_loss:
                loss.append(self.loss_func(output, label))
        # 合并成一个Tensor
        batch_output = torch.cat(batch_output, 0)
        # return        
        if if_loss:
            self.loss = self.loss_sum(loss)
            return batch_output,self.loss
        else:
            return batch_output
        
    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        
     
cnn2rnn = CNN_Linear_RNN()
decisioner = Decisioner()
    
ECG_class = ECG_Class(cnn2rnn, decisioner, PARM.ifGPU)
#%% Training Network 训练网络
if PARM.ifGPU:
    test_y = test_y.cuda()
# 迭代过程
print('Start Training')
for epoch in range(PARM.epoch): # 总体迭代次数
    for step, (x, y) in enumerate(loader):
        if PARM.ifGPU:
            y = y.cuda()
        #print('epoch:', epoch+1, ' step:', step+1)
        output, loss = ECG_class.forward(x, y, True)
        ECG_class.backward()

        # 计算准确率
        if step % 1 == 0:
            test_output = ECG_class.forward(test_x, test_y, False) # 输入测试集得到输出test_out           
            if PARM.ifGPU:
                pred_train = torch.max(output, 1)[1].cuda().data.squeeze()
                pred_test = torch.max(test_output, 1)[1].cuda().data.squeeze()
            else:
                pred_train = torch.max(output, 1)[1].data.squeeze()     # 训练集输出的分类结果
                pred_test = torch.max(test_output, 1)[1].data.squeeze() # 测试集输出的分类结果
            # 计算当前训练和测试的准确率
            accuracy_train = float((pred_train == y.data).sum()) / float(y.size(0))
            accuracy_test = float((pred_test == test_y.data).sum()) / float(test_y.size(0))
            #输出
            print('Epoch: ', epoch, ' Step', step, '| train loss: %.4f' % loss.data, '| train accuray: %.2f' % accuracy_train, '| test accuracy: %.2f' % accuracy_test)
            
            
pre=pred_test.data.cpu().numpy()+1
# pre = np.array(pred_test) + 1
test =test_y.data.cpu().numpy()+1
# test = np.array(test_y.data) + 1

pre_dict = {'Recording': [], 'Result': []}
test_dict = {'Recording': [], 'First_label': []}

count = 0
for i in range(len(pre)):
    pre_dict['Recording'].append(count)
    pre_dict['Result'].append(pre[i])

    test_dict['Recording'].append(count)
    test_dict['First_label'].append(test[i])
    count += 1

pre = pd.DataFrame(pre_dict)
test = pd.DataFrame(test_dict)
# %%
test['Second_label'] = ''
test['Third_label'] = ''
# %%
pre.to_csv('1.csv', index=False)
test.to_csv('2.csv', index=False)
score('1.csv', '2.csv')

        
