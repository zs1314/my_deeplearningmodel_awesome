"""
LeNet神经网络模型
"""
import torch
from torch import nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Sequential, Flatten
from torch.utils.tensorboard import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sequential = Sequential(
            Conv2d(3, 16, 5),
            ReLU(inplace=True),  # inplace为True节省了例外开辟内存的时间，直接覆盖原有结果，即节省内存，有节省训练时间
            MaxPool2d(2, 2),
            Conv2d(16, 32, 5),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Flatten(),  # 千万不能忘，展平，降维打击
            Linear(32 * 5 * 5, 120),
            Linear(120, 84),
            Linear(84, 10)
        )

    def forward(self, input):
        input = self.sequential(input)
        return input


# 测试模型搭建是否正确
if __name__ == "__main__":
    nn_module = LeNet()
    input = torch.ones((100, 3, 32, 32))
    output = nn_module(input)
    print(output.shape)
    writer = SummaryWriter("model_graph")
    writer.add_graph(nn_module, input)
    writer.close()
    # 搭建正确，nice


