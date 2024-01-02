import torch
# from module_save import *
# # 第一种保存方式——》相对应（用来加载模型）
# model1=torch.load("vgg16_method1.pth")
# print(model1)


# 第二种保存方式——>相对应
model2=torch.load("vgg16_method2.pth")
print(model2)

#
# # 自定义模型加载
# model3=torch.load("zs_method1.pth")
# print(model3)
# # 需要注意的点，需要引用模型文件这个模块（import）