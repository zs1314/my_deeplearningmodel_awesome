# 导入需要的库和模块
import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import torchvision.models.resnet
from model import resnet34
# 首先要明白的是我建立的模型输入尺寸为224*224，所以最基本要保证输入尺寸为224以及tensor的数据类型
# 训练数据的转化
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_data_transforms=torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomCrop(224),  # 随机裁剪成尺寸为224
        torchvision.transforms.RandomHorizontalFlip(),  # 随机翻转
        torchvision.transforms.ToTensor(),  # 转换为tensor数据类型
        torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))  # 标准化（归一化）至[0,1]1(这里主要防止梯度爆炸)
        # 注意：随机裁剪、反转等都是增大数据集数量，可以实现数据增强，其目的就是防止过拟化
        # 这里的变化由于是迁移学习，所以跟论文中的预处理方式一样，否则训练效果不好

    ]
)
# 测试数据的转化
test_data_transforms=torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        # 注意：测试集是不用进行反转之类的，只需要进行简单的尺寸变化和类型转换，以适应模型输入尺寸，因为测试集是用来测试模型的准确率
    ]
)

# 处理数据
train_dataset=ImageFolder(root='./grape_data/train',transform=train_data_transforms)
test_dataset=ImageFolder(root='./grape_data/val',transform=test_data_transforms)
print(len(train_dataset),len(test_dataset))

# 载入数据
train_loader=DataLoader(dataset=train_dataset,batch_size=36,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=test_dataset,batch_size=36,shuffle=False,num_workers=0)
# 这里的num_worker为主线程加载（windows只能为0），batch_size指的是分几批，而不是每一批有多少数据
print(len(train_loader),len(test_loader))

# 定义新的训练设备
device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

# 创建网络模型
model=resnet34()  # 迁移学习搞的是别人训练好的参数，但是还是要自己创建模型(model中创建的)，并建立，但是然后在这里直接加载别人的参数

# 加载训练好的参数
model_weight_path='resnet34-pre.pth'  # Pytorch训练好的权重参数（不是整个模型）
model.load_state_dict(torch.load(model_weight_path))

# 由于最后输出的类别有1000个，所以这里改变输出类别，添加一个全连接层
inchannel = model.fc.in_features
model.fc = nn.Linear(inchannel, 5)
model=model.to(device)

# 损失函数，这里的多分类问题仍采用交叉熵损失函数
Loss= CrossEntropyLoss(weight=None)
Loss=Loss.to(device)

# 定义tensorboard
writer=SummaryWriter(log_dir="ResNet_trend")

# 定义优化器
learning_rate=0.0002
# optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# 定义训练是用到的超参数
train_step=0
test_step=0
epoch=30   # 训练轮数
best_acc=0.0
for i in range(epoch):
    print('————————————第{}轮训练——————————————'.format(i+1))
    for data in train_loader:
        # 开始训练
        optimizer.zero_grad()  # 梯度请0，防止梯度累加，参数爆炸

        model.train()  # 这是必须的，由于有Dropout层，为了保护、固定参数
        imgs,targets=data  # 读取图片和标签
        imgs=imgs.to(device)
        targets=targets.to(device)  # 转换设备
        output=model(imgs)
        loss_single=Loss(output,targets)  # 计算误差（损失），方便下面反向传播，计算梯度
        # 优化（反向传播、梯度下降）
        loss_single.backward()  # 反向传播，计算梯度
        optimizer.step()  # 优化，更新模型参数
        train_step += 1
        # 打印数据
        if train_step % 75==0:
            print("训练次数：%d   loss_single：%f" % (train_step, loss_single))
        # 本轮训练结束

    total_test_loss = 0
    total_test_accuracy = 0
    # 针对本轮的训练，对测试集进行测试（验证），查看准确率
    model.eval()
    with torch.no_grad():  # 关闭梯度，测试时是不能更新参数的，所以也就不需要梯度，为避免梯度在测试中被污染，直接关闭
        # 以下步骤与训练基本一致，但不要优化，也就不需要计算梯度、反向传播之类
        for test_data in test_loader:
            test_imgs, test_targets = test_data
            test_imgs = test_imgs.to(device)
            test_targets = test_targets.to(device)
            output_test = model(test_imgs)
            loss_test_single = Loss(output_test, test_targets)
            total_test_loss += loss_test_single  # 统计一轮内的总损失

            # 注意：output_test类型为一维张量，分别为每个类别的非线性概率
            accuracy = (output_test.argmax(1) == test_targets).sum()
            total_test_accuracy += accuracy

    # 打印数据
    average_accuracy=total_test_accuracy / len(test_dataset)
    print("经过第{}轮训练   Loss：{}".format(i + 1, total_test_loss))
    print("经过第{}轮训练   准确率：{}".format(i + 1, average_accuracy ))

    # add_scalar绘制图像
    writer.add_scalar(tag="Loss", scalar_value=total_test_loss, global_step=i + 1)
    writer.add_scalar(tag="Accuracy", scalar_value=(total_test_accuracy / len(test_dataset)), global_step=i + 1)

    # 保存模型,便于后面demo直接加载调用模型
    if average_accuracy>best_acc:
        best_acc=average_accuracy
        torch.save(model.state_dict(),"ResNet_method.pth")  # 保留模型的超参数
        # 这里保存的是经过迁移学习后得到，在进行一定训练后得到的参数，后面预测就引用这个pth
        print("第{}轮模型保存成功".format(i + 1))
