from torch.utils.tensorboard import SummaryWriter
from my_numpy import demo_numpy as np
from PIL import Image

writer=SummaryWriter("logs")
img_path= "/temp/data/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)
print(img_array.shape)
writer.add_image("test",img_array,2, dataformats='HWC')


# y=x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
# scalar——标量

writer.close()