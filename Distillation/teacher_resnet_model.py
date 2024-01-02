"""
特点：
1、采用了残差结构
2、用了BN
这个模型我五个（18、34、50、101、152都搭建了）
搭建是有个技巧：只有深度大的三个大模型，才有stride=2，可以根据这个来搭建，区分这五个
"""
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU


# 残差结构(这里为18、34层的ResNet的,50,101,152的与这个略微不同，有三层)
class BasicBlock(nn.Module):
    expansion = 1  # 这里代表的是中间的卷积的个数有没有发生变化，即原论文中，残差有无出现虚线（即需要改变原输入的shape）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        #  downsample表示是否有那个虚线，用来过渡上一个残差的输出与这个的输入的格式不一致
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=stride, padding=1, bias=False)
        """
        当stride为1时，就是实线结构，output=(input-3+2)/1+1=output，输入和输出的形状不变
        当stride=2时，就是虚线结构——为了衔接上一层的输入和这一层的输出一致，output=(input-3+2)/2+1=input/2+0.5=input/2 (高和宽减半)
        这里的偏置之所以不设，是因为经过BN，设于不设结果都相同
        """
        self.bn1 = BatchNorm2d(out_channels)
        # BN位于卷积层和非线性激活中间
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False)
        # 这里的卷积输入的通道数就为上一层输出的通道数，由于是残差，正道和偏道的形状是相同的，所以正道不管怎么变，最后输出的就是输入的
        self.bn2 = BatchNorm2d(out_channels)
        self.downsample = downsample  # 下采样，经过下采样后，就需要搞虚线的残差结构

    def forward(self, x):
        identity = x  # 偏道的输入，也是输出（若是实线，即没有下采样）
        if self.downsample is not None:
            identity = self.downsample(x)  # 若是虚线，则要更新一下输入，保证最后两个路径输出的形状是相同的

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 注意：先要正道和偏道加和后再非线性激活，这里的加和就是普通的相加，不是像VGG一样在维数上的叠加
        out += identity  # 两个路径相加
        out = self.relu(out)
        return out


# 残差结构（50、101、152的ResNet的）
class Bottleneck(nn.Module):
    expansion = 4
    """
    50,101,152的残差有三层，且最后一层的卷积核的个数为前面两层的个数的4倍
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        # 第一层卷积有虚线和无虚线的都是一样的结构
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                            bias=False, padding=1)
        # 这里的stride实线为1，虚线为2，所以把参数赋值给了变量，
        # 实线：output=(input-3-2)/1+1=input  虚线：output=(input-3+2)/2+1=input/2
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                            stride=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 即有虚线（也有下采样）
            identity = self.downsample(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(x)
        output = self.conv3(output)
        output = self.bn3(output)

        output += identity
        output = self.relu(output)
        return output


# 整个网络结构
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
