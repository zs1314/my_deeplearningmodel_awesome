import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,batch_size=100)


class nn_maxpool(nn.Module):
    def __init__(self):
        super(nn_maxpool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


writer = SummaryWriter('my_maxpool')
step = 0
my_maxpool = nn_maxpool()
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = my_maxpool(imgs)
    writer.add_images('output', output, step)
    step += 1
