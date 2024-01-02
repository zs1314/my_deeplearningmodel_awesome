from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#pyhton 的用法  ->tensor数据类型
"""
通过 transforms.Totensor认识两个问题
1、transforms该如何使用
2、为什么需要使用Tensor数据类型
"""
img_path= r"/dataset/train/ants/0013035.jpg"
img=Image.open(img_path)
writer=SummaryWriter("logs")

tensor_trains=transforms.ToTensor()
# tensor_trains这里为transforms.ToTensor类的一个对象
# 其中有一个内置方法__call__，即调用对象时可以像调用方法一样

tensor_img=tensor_trains(img)
print(type(tensor_img))
print(tensor_img)

writer.add_image("Tensor",tensor_img)
writer.close()