import torch
from torchsummary import summary
from my_swish_mobilenet_model import CustomMobileNetV2  # 导入你自定义的模型

from teacher_resnet_model import resnet34

# # 创建自定义MobileNetV2模型实例
# custom_mobilenet = CustomMobileNetV2(num_classes=4)  # 假设有4个类别
#
# # 打印模型结构摘要
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# custom_mobilenet.to(device)
# summary(custom_mobilenet, (3, 224, 224))  # 输入图像大小为(3, 224, 224)

custom_resnet=resnet34(num_classes=4)

# 打印模型结构摘要
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_resnet.to(device)
summary(custom_resnet, (3, 224, 224))  # 输入图像大小为(3, 224, 224)

