import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class nn_CAF(nn.Module):
    def __init__(self):
        super(nn_CAF, self).__init__()
        # self.conv1=Conv2d(3,32,5,padding=2)
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2=MaxPool2d(2)
        # self.conv3=Conv2d(32,64,5,padding=2)
        # self.maxpool3=MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)
        # 模型的集成
        self.sequential=Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.sequential(x)
        return x


writer=SummaryWriter('nn_CAF')
module=nn_CAF()
print(module)
input=torch.ones((64,3,32,32))
output=module(input)
writer.add_graph(module,input)
writer.close()
print(output.shape)

