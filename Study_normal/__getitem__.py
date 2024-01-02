import os

from torch.utils.data import Dataset
from PIL import Image


class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        img = Image.open(image_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


root_dir = "dataset/train"
ants_label_dir="ants"
bees_label_dir="bees"
ant_dataset = Mydata(root_dir,ants_label_dir)
bee_dataset=Mydata(root_dir,bees_label_dir)

train_dataset=ant_dataset+bee_dataset
