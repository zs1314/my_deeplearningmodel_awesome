import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

class nn_sigmoid(nn.Module):
    def __init__(self):
        super(nn_sigmoid, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

class nn_relu(nn.Module):
    def __init__(self):
        super(nn_relu, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


writer=SummaryWriter('my_relu')
step=0
relu1 = nn_relu()
for data in dataloader:
    imgs, targets = data
    writer.add_images('input',imgs,step)
    output=relu1(imgs)
    writer.add_images('output',output,step)

writer.close()
