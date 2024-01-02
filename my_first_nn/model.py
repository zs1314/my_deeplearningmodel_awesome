"""
搭建神经网络
"""
import torch
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear


class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        # 开始搭建神经网络内部的层
        self.modle = Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)  # 有十个类别，最后线性层（全连接）输出chanel为10
        )

    def forward(self, x):
        x = self.modle(x)
        return x


# 用于检测神经网络是否搭建正确，且只在这个文件内运行，才运行下列代码，对于模型的训练、预测等无影响
if __name__ == "__main__":
    my_model = nn_model()
    input = torch.ones((100, 3, 32, 32))
    output = my_model(input)
    print(output.shape)
# 搭建正确
