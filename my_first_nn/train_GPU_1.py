"""用GPU训练"""
from torch.utils.tensorboard import SummaryWriter

"""
神经网络模型、数据(输入，标签)、损失函数可用CUDA训练（用.cuda）
"""

# 导入模块

import torch.nn
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from model import *

# 下载数据集，并保存在文件中
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)  # 训练数据集
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)  # 测试数据集
# 可以看一下数据集的长度
print("训练数据集长度 %s" % len(train_data))
print("测试数据集的长度{}".format(len(test_data)))

# 加载数据集，并把数据分成多份，每一份数量为batch_size
train_dataloader = DataLoader(dataset=train_data, batch_size=100)  # 训练数据集
test_dataloader = DataLoader(dataset=test_data, batch_size=100)  # 测试数据集

print(len(test_dataloader))
# 创建网络模型
nn_moduel = nn_model()
nn_moduel = nn_moduel.cuda()

# 损失函数
# 这是一个分类问题，可以用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 定义优化器
learning_rate = 0.01  # 学习速率，后期进行调整
optimizer = torch.optim.SGD(nn_moduel.parameters(), lr=learning_rate)  # 采用随机梯度下降

# tensorboard的使用，更形象
writer=SummaryWriter("my_first_nn")

# 设置训练网络的参数
train_step = 0  # 训练的次数
test_step = 0  # 测试的次数
epoch = 50  # 训练的轮数

for i in range(epoch):
    # 将训练分为几轮，更直观感受优化
    print("——————————第{}轮训练————————".format(i + 1))
    # 开始训练
    nn_moduel.train()  # 有些特殊的层需要调用这个
    for data in train_dataloader:  # 以一次batch_size取出图片和标签
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        # 传入图片，得到训练的预测值
        output = nn_moduel(imgs)
        # 通过损失函数,传入训练得到的标签与真正的标签，计算误差
        loss_value = loss_fn(output, targets)

        """优化器优化模型"""
        # 将梯度至0，防止梯度累加
        optimizer.zero_grad()
        # 得到本次训练梯度
        loss_value.backward()
        # 通过梯度下降，以及优化器，优化参数，减小损失
        optimizer.step()
        # 训练次数+1
        train_step += 1
        if train_step % 100 == 0:
            print("训练次数：%d   Loss：%lf" % (train_step, loss_value.item()))  # item可以把tensor数据类型转换为真实的数字
            # 注意：训练改变的是权重和偏置

    #  测试步骤，更直观测试优化是否有效(看每一轮训练优化参数后，对测试集上的数据预测损失是否减小，即是否真正优化)
    nn_moduel.eval()  # 同上rain，可以固定参数，防止验证或测试时，改变参数（权重和偏置）
    total_test_loss = 0  # 整体上（一轮）的损失
    total_test_accuracy = 0  # 整体上（一轮）的正确率
    with torch.no_grad():
        # 进入的前提为关闭梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output_test = nn_moduel(imgs)
            loss_test = loss_fn(output_test, targets)
            accuracy = (output_test.argmax(1) == targets).sum()
            total_test_loss += loss_test.item()
            total_test_accuracy += accuracy

    print("第{}轮整体测试集上的Loss：{}".format(i + 1, total_test_loss))
    print("第{}轮整体测试集的正确率:{}".format(i + 1, total_test_accuracy / len(test_data)))
    writer.add_scalar("Loss",scalar_value=total_test_loss,global_step=i+1)
    writer.add_scalar("accuracy",scalar_value=total_test_accuracy,global_step=i+1)
torch.save(nn_moduel,"zs_method{}.pth".format(epoch))
print("保存成功")