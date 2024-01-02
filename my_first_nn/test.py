"""demo或者验证集"""
import torch
import torchvision
from PIL import Image

# 引入要验证（预测）的图片
img_path= "dog.jpg"
img=Image.open(img_path)
img=img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）

# 变换图片格式，以此适应神经网络的进入格式
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
img=transform(img)
print(img.shape)
img=torch.reshape(img,(1,3,32,32))
img=img.cuda()
# 加载网络模型
model=torch.load("zs_method50.pth")
# 开始验证
model.eval()
with torch.no_grad():
    output=model(img)
print(output)
labels=['飞机','汽车','鸟','猫','鹿','狗','蛙','马','船','火车']
print(output.argmax(1))
print("判断为:",labels[output.argmax(1)])



