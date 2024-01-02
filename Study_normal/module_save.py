import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

# pretrain为False代表模型中的参数都是初始化，未经过训练的,不是没有参数
vgg16=torchvision.models.vgg16(pretrained=False)

# 第一种保存方式 (不仅保留了网络模型的结构，还保留了模型中的参数)  模型结构+初始化参数
torch.save(vgg16,"vgg16_method1.pth")

# 第二中保存方式 (只保留了参数)   模型参数
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

# 自定义模型
class Zs(nn.Module):
    def __init__(self):
        super(Zs, self).__init__()
        self.conv1=Conv2d(3,6,kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x


zs=Zs()
torch.save(zs,"zs_method1.pth")
