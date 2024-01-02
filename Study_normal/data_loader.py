import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 准备的测试集，量小，测试集是给平常测试用，而训练集量大，给机器学习训练神经网络用的
test_data=torchvision.datasets.CIFAR10("torchvision_dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
# dataset中__getitem__返回值为image(tensor类型),target（数据集中标签）

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

# 测试数据集中第一张图片及target（标签）---即图片的类容是什么（一个分类）
image,target=test_data[0]
print(image.shape)
print(target)

writer=SummaryWriter("dataloader")
for epcoh in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        # 对test_loader中是数据进行打包，batch_size=4意思是每批次加载4个图片及他的标签，再进行打包，imgs中包含4张图片，target中包含这四张图片的标签
        # print(imgs_predict.shape)
        # print(targets)
        writer.add_images("test_data %d"% epcoh,imgs,step)
        step+=1
writer.close()
