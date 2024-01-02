import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1)

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


# 交叉熵损失
loss=nn.CrossEntropyLoss()
my_module=nn_CAF()

for data in dataloader:
    imgs,targets=data
    output=my_module(imgs)
    # print(output)
    # print(targets)

    # 这里得到误差
    result=loss(output,targets)
    print(result)
    # 这里通过反向传播，得到梯度，通过优化器对梯度的利用，对参数进行调整，最后使误差降低达到优化的目的
    result.backward()