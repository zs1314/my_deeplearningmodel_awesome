# 导入库和模块
import torch.optim
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 对数据进行格式变换
transform = torchvision.transforms.Compose(
    [  # 转换为tensor数据类型
        torchvision.transforms.ToTensor(),
        # 标准化处理
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# 定义新的训练设备
device = torch.device("cuda")
print(device)

# 下载数据
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)  # 训练集
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)  # 测试集

# 加载数据
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

# 创建LeNet网络模型
lenet = LeNet()
lenet = lenet.to(device)

# 计算误差（损失），用交叉熵损失函数计算，这里知识定义，后面训练会调用，并用其计算
loss = CrossEntropyLoss(weight=None)
loss = loss.to(device)

# 定义优化器，这里算法采用随机梯度下降
learning_rate = 0.01
optimizer = torch.optim.SGD(params=lenet.parameters(), lr=learning_rate)

# tensorboard的使用，更直观
writer = SummaryWriter("LeNet")  # 定义日志

# 开始定义一些训练网络的参数
train_step = 0
test_step = 0
epoch = 300  # 训练轮数
for i in range(epoch):
    print("————————————第{}轮训练——————————".format(i + 1))
    for data in train_loader:
        # 开始训练

        lenet.train()  # 有些特殊层需要这个，目的：保护参数不受影响，固定参数
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = lenet(imgs)
        loss_single = loss(output, targets)  # 通过训练得到的标签与实际的标签，来通过交叉熵损失函数计算误差

        # 开始优化
        optimizer.zero_grad()  # 梯度置0，防止梯度累加
        loss_single.backward()  # 反向传播，计算梯度
        optimizer.step()  # 通过梯度下降，优化器，更新参数，减小误差
        train_step += 1
        if train_step % 100 == 0:
            print("训练次数：%d   loss_single：%f" % (train_step, loss_single))

    """ 开始通过对训练集误差、准确率进行分析，预测（每一轮搞一次）"""
    #         关闭梯度，防止影响梯度，进而影响参数(这里不用进行优化，只是进行一个正确率和损失的计算和展示)
    with torch.no_grad():
        # 基本步骤同上述训练一致
        total_test_loss = 0.0
        total_accuracy = 0.0
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output_test = lenet(imgs)
            loss_test_single = loss(output_test, targets)
            total_test_loss += loss_test_single
            # 注意：output_test类型为一维张量，10个数据，分别为每个类别的非线性概率
            accuracy = (output_test.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("经过{}轮训练，Loss为{}".format(i + 1, total_test_loss))
    print("经过{}轮训练,正确率为{}".format(i + 1, total_accuracy / len(test_loader)))

    # add_scalar绘制图像
    writer.add_scalar(tag="Loss", scalar_value=total_test_loss, global_step=i + 1)
    writer.add_scalar(tag="Accuracy", scalar_value=(total_accuracy / len(test_loader)), global_step=i + 1)
    # writer.close()
    # 保存模型，方便调用。后续可根据tensorboard绘制图像或控制台显示的正确率确定哪个模型最好（损失最少，正确率最高）
    torch.save(lenet, "LeNet_method_{}.pth".format(i + 1))
    print("第{}轮模型保存成功".format(i + 1))
