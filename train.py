#这个模型可以训练加测试一条龙，输出的最终结果是MSE结果
#目前还没有在这台机器上运行过，如果有人运行并发现有问题，麻烦改动后踢一下王拓源，让我再本地也同步一下正确的代码
#git已经在学了，等我学会了就把上面的注释删掉

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd

import model as md

#该函数希望输入的数据是两个tenser矩阵，其中一部分是数据，另一部分是对应的正确光谱
def train(data,label):
    device = torch.device("cuda")
    lamba_l2=0.01#用于控制l2正则化强度的超参数
    train_times=300#用于控制训练次数
    model=md.DNN_Net().to(device)
    data_train,data_test,label_train,label_test=train_test_split(data,label,test_size=0.2)#80%训练集
    train_set=TensorDataset(data_train,label_train)
    test_set=TensorDataset(data_test,label_test)
    dataload_train=DataLoader(train_set,batch_size=64,shuffle=True)
    dataload_test=DataLoader(test_set,batch_size=64,shuffle=True)
    model.train()
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for i in range(train_times):
        for inputs,labels in dataload_train:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            optimizer.zero_grad()
            loss=criterion(outputs,labels)
            l2_reg=sum(param.pow(2).sum() for param in model.parameters())
            total_loss=loss+l2_reg*lamba_l2
            total_loss.backward()
            optimizer.step()
        print(f'训练次数{i}/{train_times}')
    model.eval()
    loss_trial=0
    times=0
    with torch.no_grad():
        for inputs,labels in dataload_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss_trial=loss_trial+loss
            times=times+1
    print(f'最终的损失评价(MSE)是:{loss_trial/times}')
    return

if __name__=='__main__':
    data=pd.read_excel('data.xlsx')
    label=pd.read_excel('label.xlsx')
    data=torch.tensor(data, dtype=torch.float32)
    label=torch.tensor(label, dtype=torch.float32)
    train(data,label)