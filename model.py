import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_Net(nn.Module):
    def __init__(self):
        super(DNN_Net, self).__init__()
        self.fc1=nn.Linear(100,500)
        self.fc2=nn.Linear(500,1000)
        self.fc3=nn.Linear(1000,1500)
        self.fc4=nn.Linear(1500,500)
        self.fc5=nn.Linear(500,2000)
        self.fc6=nn.Linear(2000,1000)
        self.fc7=nn.Linear(1000,500)
        self.fc8=nn.Linear(500,165)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=F.relu(self.fc7(x))
        x=F.relu(self.fc8(x))
        return x

if __name__ == '__main__':
    net=DNN_Net()
    print(net)