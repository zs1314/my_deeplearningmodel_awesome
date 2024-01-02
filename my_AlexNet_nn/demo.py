"""
demo
"""

import torch
import torchvision
from PIL import Image


class predict:

    def my_predict(self, img_path, model_method):
        # 引入图片
        img = Image.open(img_path)  # 将图片转换为PIL形式,方便转换为tensor数据类型

        # 图片转换
        img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                     torchvision.transforms.ToTensor()]
                                                    )
        img = transforms(img)
        img = torch.reshape(img, (1, 3, 224, 224))  # 转换为适合进入神经网络的类型格式
        img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

        # 标签（三个类别）
        labels = ['茶云纹','茶白星病','茶赤叶斑病']

        # 建立模型,加载模型(训练得到的，选误差最小的)
        my_lenet = torch.load(model_method)
        my_lenet(img)

        # 开始预测
        my_lenet.eval()
        with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
            output = my_lenet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
            predict = torch.softmax(output, dim=1)
            predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）
            print("预测结果为：", labels[output.argmax(1)])
            print("概率为：", predict[0][output.argmax(1)].numpy().item())  # item()可以直接显示数值

        return labels[output.argmax(1)]


# 测试是否正常
if __name__ == "__main__":
    pre = predict()
    pre.my_predict(r"D:\pythonProject1\my_AlexNet_nn\img_4.png",r"D:\pythonProject1\my_AlexNet_nn\model_method\AlexNet_method4.pth")
