"""
transforms是对图片进行一些处理，实际上为一个py文件，包含许多类，可以import后调用
"""
import torchvision.transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
img_path="dataset/train/ants/5650366_e22b7e1065.jpg"
img=Image.open(img_path)
print(img)
# 注意:Image.open()是Python的一个内置方法,参数为图片路径,用来打开图片
# 且格式为PIL


#Totensor 将PIL的类型转换为Tensor
trans_tensor=transforms.ToTensor()
img_tensor=trans_tensor(img)
writer.add_image("To_tensor",img_tensor,3)
writer.close()
print(img_tensor)
# 此时img_tensor的数据类型为tensor类型


# Normalize  ---归一化
print(img_tensor[0][0][0])
trans_form=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_form(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,3)
writer.close()
# 注意:一定要关闭writer,否则tensorboard显示不出来图像

# Resize
# 把一个照片的尺寸改为指定的尺寸
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_tensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)
writer.close()


# Compose  --其实就是多种图形变换的组合(串联多种图形变换)
trans_resize2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize2,trans_tensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize2",img_resize_2,1)

# RandonCrop ---随机裁剪
trans_random=transforms.RandomCrop(200,400)
trans_compose_2=transforms.Compose([trans_random,trans_tensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)
writer.close()