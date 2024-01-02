"""
VGG网络特点
1、通过堆叠两个3*3卷积核来替代5*5卷积核，堆叠三个3*3的卷积核来替代7*7的卷积核
——>减少参数，获得相同的感受野（但VGG参数量还是非常大，臃肿）——>注意：该模型中卷积核大小均为3*3
2、卷积层——>使H、W不变，通道数变为卷积核个数
   最大池化层——>使HW减半，通道数不变
   （卷积和全连接都要加非线性激活，保证函数的非线性）
"""
# 导入模块
import torch
from torch import nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Dropout, Linear, Flatten


class Vgg16_net(nn.Module):
    def __init__(self, num_class,init_weight=True):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_class)
        )
        # 参数初始化
        if init_weight:  # 如果进行参数初始化
            for m in self.modules():  # 对于模型的每一层
                if isinstance(m, nn.Conv2d):  # 如果是卷积层
                    # 使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):  # 如果是线性层
                    # 正态初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    nn.init.constant_(m.bias, 0)

    # 正向传播
    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1
        # 如果出现x.size(0)表示的是batchsize的值
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 建立模型
if __name__ == "__main__":
    model = Vgg16_net(5)
    # input = torch.ones((1, 3, 224, 224))
    # print(input.shape)
    # output = model(input)
    # print(output.shape)
    print(model)
