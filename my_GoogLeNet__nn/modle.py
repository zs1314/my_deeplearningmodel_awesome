"""
模型特点：
1、Inception结构
2、1*1卷积核——>降维（不会改变特征层的高与宽）
3、采用平均池化
4、有辅助分类器（两个）
"""

# 此为改变后的模型
# # 导入模块与库
# import torch
# from torch import nn
# from torch.nn import Conv2d, ReLU, MaxPool2d, AvgPool2d, Linear
# import torch.nn.functional as F
#
#
# class GogoLeNet(nn.Module):
#     def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):  # nun_class表示分类数目，aux_logits表示是否用辅助分类器
#         # init_weights表示是否初始化参数——>对卷积层和全连接层初始化
#         super(GogoLeNet, self).__init__()
#         self.aux_logits = aux_logits  # 是否使用辅助分类
#
#         self.conv1 = Basic_Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)  # ceil_mode=True表示有余数时，向上取整
#
#         self.conv2 = Basic_Conv2d(64, 64, kernel_size=1)
#         self.conv3 = Basic_Conv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)
#
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#
#         self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#
#         if aux_logits:  # 是否需要辅助分类
#             self.aux1 = InceptionAux(512, num_classes)  # 第一个
#             self.aux2 = InceptionAux(528, num_classes)  # 第二个
#
#         self.avgpool = nn.AdaptiveMaxPool2d((1, 1))  # 自适应平均池化层
#         self.dropout = nn.Dropout(p=0.4)
#         self.fc = Linear(1024, num_classes)
#
#         if init_weights:
#             self._initialize_weigth()  # 是否初始化权重
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.maxpool2(x)
#         x = self.inception3a
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#         x = self.inception4a(x)
#         if self.training and self.aux_logits:  # 第一个辅助分类，还要看是否处于训练中
#             aux1 = self.aux1(x)
#
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#
#         if self.training and self.aux_logits:  # 第二个辅助分类
#             aux2 = self.aux2()
#
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)  # 展平（在通道那一维）
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         if self.training and self.aux_logits:  # 若是在训练中且需要辅助分类器，则返回辅助分类器的预测值
#             return x, aux2, aux1
#         return x
#
#     def _initialize_weigth(self):
#         # 参数初始化
#         for m in self.modules():  # 对于模型的每一层
#             if isinstance(m, nn.Conv2d):  # 如果是卷积层
#                 # 使用kaiming初始化
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 # 如果bias不为空，固定为0
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):  # 如果是线性层
#                 # 正态初始化
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 # bias则固定为0
#                 nn.init.constant_(m.bias, 0)
#
#
# # Inception结构在GogoLeNet中都是一样，然后封装
# class Inception(nn.Module):
#     """
#           注意：
#           1、所有分支的输入通道数都是一样的
#           2、每个分支所得到的feature_map的高宽都是一样的，便于在深度哪一维叠加
#     """
#
#     def __init__(self, in_channels, ch1x1_conv, ch3x3_reduce, ch3x3_conv, ch5x5_reduce, ch5x5_conv, pool_project):
#         super(Inception, self).__init__()
#         # 第一个分支
#         self.branch1 = Basic_Conv2d(in_channels, ch1x1_conv, kernel_size=1)  # padding默认为1
#
#         # 第二个分支
#         self.branch2 = nn.Sequential(
#             Basic_Conv2d(in_channels, ch3x3_reduce, kernel_size=1),  # reduce的输出是正常卷积的输入
#             Basic_Conv2d(ch3x3_reduce, ch3x3_conv, kernel_size=3, padding=1)
#         )
#
#         # 第三个分支
#         self.branch3 = nn.Sequential(
#             Basic_Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
#             Basic_Conv2d(ch5x5_reduce, ch5x5_conv, kernel_size=5, padding=2)  # 保证输入的和输出矩阵HW的相等
#         )
#
#         # 第四个分支
#         self.branch4 = nn.Sequential(
#             MaxPool2d(kernel_size=3, stride=1, padding=1),  # 保证输入特征矩阵的高和宽于输出的一样，设置特定的stride、padding(通过公式计算)
#             Basic_Conv2d(in_channels, pool_project, kernel_size=1)  # 最大池化是不改变通道数的
#         )
#
#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)  # cat是将四个分支组合在一起，1代表以index=1(深度/通道)的维度组合（[batch_size,channel,heigth,width]）
#
#
# # 辅助分类器
# class InceptionAux(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.AvgPool = AvgPool2d(kernel_size=5, stride=3)
#         self.conv = Basic_Conv2d(in_channels, 128, kernel_size=1)
#
#         # 全连接层
#         self.fc1 = Linear(2048, 1024)
#         self.fc2 = Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = self.AvgPool(x)
#         x = self.conv(x)
#         x = torch.flatten(x, 1)  # 展平
#         # 开始全连接
#         x = F.dropout(x, p=0.5, training=self.training)  # 随机失活，泛化特征  training=self.training表示model.train/.eval中起作用
#         x = self.fc1(x)
#         x = F.relu(x, inplace=True)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.fc2(x)
#         return x
#
#
# # 封装一个卷积块（一个卷积层+非线性激活）
# class Basic_Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):  # **kwargs表示关键字参数，这里表示以字典形式传入
#         super(Basic_Conv2d, self).__init__()
#         self.conv = Conv2d(in_channels, out_channels, **kwargs)
#         self.ReLU = ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.ReLU(x)
#         return x

import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    input = torch.ones((1, 3, 224, 224))
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    print(net)
    output = net(input)
    print(output)
