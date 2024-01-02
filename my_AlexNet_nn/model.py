"""
完成对AlexNet的模型搭建
"""

# 导入模块和库
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Dropout, Linear
class AlexNet(nn.Module):
    def __init__(self,classes_num):
        super(AlexNet, self).__init__()
        self.sequential=Sequential(
            # 其实有些默认参数没必要依次写出来，但我为了加深理解，所以把每个参数都写出来了
            # 特征提取
            Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2),
            ReLU(inplace=True),  # inplace增加计算量，但减少内存的占用 （每卷积一次，非线性激活一次，给函数引入非线性）
            MaxPool2d(kernel_size=3,stride=2,padding=0),  # 最大池化其实不会改变通道数，会改变图片的尺寸，所以这里的参数没有通道（深度）
            Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3,stride=2,padding=0),
            Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=192,out_channels=128,kernel_size=3,padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3,stride=2,padding=0),

            # 展平
            Flatten(),  # 这个是必须的，观察模型的结构可以看出，卷积层与全连接层之间，是要展平成一维的张量（一维向量）的！！！
                        # 是从三个维度展平的（CHW）,batch_size是不去动的

            # 开始分类（通过线性）
            Dropout(p=0.5),  # 这是AlexNet的一个创新点，目的防止过拟合，采用以p的概率随机失活神经元（这里的p其实也算个超参数，也可以自己调整调整）
            Linear(in_features=4608,out_features=2048),  # 经过上面展平后，节点数是有4608个（可以计算的128*6*6(CHW)）
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(in_features=2048,out_features=2048),  # 注意：这里的输入之所以没减半，是因为Dropout随机失活，只是失活，但没有消失(这也是个易错点)
            ReLU(inplace=True),
            Linear(in_features=2048,out_features=classes_num)
            # 注意每层都要激活操作，因为：激活是非线性的，如果不激活，每层就是纯线性的变换，连续多层线性和一层是等效的，那么则神经网络的”深度“就不起效果了
        )

    # 正向传播
    def forward(self,x):
        x=self.sequential(x)
        return x
