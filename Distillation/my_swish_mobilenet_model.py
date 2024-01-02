# import torch
# import torch.nn as nn
# from torchvision.models import mobilenet_v2
#
#
# # 自定义激活函数
# class MySwishActivation(nn.Module):
#     def forward(self, input):
#         return input * torch.sigmoid(input)/6
#
#
# # 自定义MobileNetV2模型，替换所有激活函数为自定义激活函数
# class CustomMobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(CustomMobileNetV2, self).__init__()
#         # 加载MobileNetV2预训练模型
#         self.mobilenet_v2 = mobilenet_v2(pretrained=True)
#
#         # 替换所有激活函数为自定义激活函数
#         self.replace_relu_with_swish(self.mobilenet_v2.features)
#
#         # 修改分类层以适应特定的任务
#         self.mobilenet_v2.classifier[1] = nn.Linear(1280, num_classes)
#
#     def replace_relu_with_swish(self, module):
#         for name, child in module.named_children():
#             if isinstance(child, nn.ReLU6):
#                 setattr(module, name, MySwishActivation())
#             else:
#                 self.replace_relu_with_swish(child)
#
#     def forward(self, x):
#         return self.mobilenet_v2(x)
#
#
# # 示例用法
# if __name__ == '__main__':
#     # 创建自定义MobileNetV2模型实例
#     custom_mobilenet = CustomMobileNetV2(num_classes=4)  # 假设有4个类别
#
#     # 构造一个输入张量
#     input_tensor = torch.randn(1, 3, 224, 224)  # 假设输入图像大小为224x224
#
#     # 使用模型进行前向传播
#     output_tensor = custom_mobilenet(input_tensor)
#
#     # 打印输出
#     print("Input: ", input_tensor.size())
#     print("Output: ", output_tensor.size())
#     print(custom_mobilenet)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


# 自定义激活函数
class MySwishActivation(nn.Module):
    def forward(self, x):
        return x * F.relu6(x+3) / 6


# 自定义MobileNetV2模型，替换所有激活函数为自定义激活函数
class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomMobileNetV2, self).__init__()
        # 加载MobileNetV2预训练模型
        self.mobilenet_v2 = mobilenet_v2(pretrained=True)

        # 替换所有激活函数为自定义激活函数
        self.replace_activation(self.mobilenet_v2)

        # 修改分类层以适应特定的任务
        self.mobilenet_v2.classifier[1] = nn.Linear(1280, num_classes)

    def replace_activation(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU6):
                setattr(module, name, MySwishActivation())
            else:
                self.replace_activation(child)

    def forward(self, x):
        return self.mobilenet_v2(x)


# 示例用法
if __name__ == '__main__':
    # 创建自定义MobileNetV2模型实例
    custom_mobilenet = CustomMobileNetV2(num_classes=4)  # 假设有10个类别

    # 构造一个输入张量
    input_tensor = torch.randn(1, 3, 224, 224)  # 假设输入图像大小为224x224

    # 使用模型进行前向传播
    output_tensor = custom_mobilenet(input_tensor)

    # 打印输出
    print("Input: ", input_tensor.size())
    print("Output: ", output_tensor.size())
    print(custom_mobilenet)
    #
    # with torch.no_grad():
    #     torch.onnx.export(
    #         custom_mobilenet,  # 要转换的模型
    #         input_tensor,  # 模型的任意一组输入
    #         'custom_mobilenet.onnx',  # 导出的 ONNX 文件名
    #         opset_version=11,  # ONNX 算子集版本
    #         input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
    #         output_names=['output']  # 输出 Tensor 的名称（自己起名字）
    #     )
    #
    # import onnx
    # # 读取 ONNX 模型
    # onnx_model = onnx.load('custom_mobilenet.onnx')
    #
    # # # 检查模型格式是否正确
    # # onnx.checker.check_model(onnx_model)
    # #
    # # print('无报错，onnx模型载入成功')
    #
    # # 以可读的形式打印计算图
    # print(onnx.helper.printable_graph(onnx_model.graph))