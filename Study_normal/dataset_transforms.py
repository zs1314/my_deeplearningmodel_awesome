# 主要从Torchvision中下载数据集使用
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root="torchvision_dataset",transform=dataset_transforms,train=True,download=True)
# 训练集(按照train来区分),transforms以此来对dataset中所有图片进行处理
test_set=torchvision.datasets.CIFAR10(root="torchvision_dataset",transform=dataset_transforms,train=False,download=True)
# 测试集
# print(test_set[0])
# print(train_set.classes)
# img,target=test_set[0]
# # target是标签
# print(img,target)
# img.show()

print(test_set[0])
writer=SummaryWriter("Torchvision")
# 这里的参数实际上就是日志的名字，随便取
for i in range(10):
    image,target=test_set[i]
    writer.add_image("test_ste",image,i)
writer.close()