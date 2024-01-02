"""
改代码演示卷积层的基本操作

"""
from tkinter import Variable

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)


class my_conv(nn.Module):
    def __init__(self):
        super(my_conv, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter('my_nn_conv')
conv1 = my_conv()
print(dataset.data.shape)
print(conv1)
# 以batch_size为一包从dataset中抽取数据集
step=0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input',imgs,step)
    # print(inputs.shape)
    output = conv1(imgs)
    # print(output.shape)
    # print(output.shape)
    outputs = torch.reshape(imgs, (-1, 3, 32, 32))
    writer.add_images('output',output,step)
    step+=1
