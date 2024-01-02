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

# 根据不同的需求，找到相应的优化器算法
# lr指学习率，学习的快慢（一般从大到小），太多，会造成不稳定，太小，会导致训练时间过长
optim=torch.optim.SGD(my_module.parameters(),lr=0.01)

for epcoh in range(20):
    run_loss=0.0
    for data in dataloader:
        imgs,targets=data
        output=my_module(imgs)
        # print(output)
        # print(targets)

        # 这里得到误差
        result_loss=loss(output,targets)

        # 这里是将梯度设置为0，若果没有这步，那梯度会一直累加，权重直接爆炸（上一次的梯度对这一次无任何作用，分多批次进行梯度下降）
        optim.zero_grad()

        # 这里通过反向传播，得到每一个结点梯度，通过优化器对梯度的利用，对权重的参数进行调整，最后使误差降低达到优化的目的
        result_loss.backward()

        # 调用优化器，开始对每个参数进行调优
        optim.step()

        # 对每一轮的误差进行累加，可以更直观的看到通过优化器后，误差的减小
        run_loss+=result_loss
    print(run_loss)