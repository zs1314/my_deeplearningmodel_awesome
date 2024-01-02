import matplotlib.pyplot as plt
from PIL import Image
import torch

from ResNet_ui_nn.model import resnet34

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 导入pillow的中文字体（opencv没有中文）
from PIL import ImageFont, ImageDraw

font = ImageFont.truetype('SimHei.ttf', 50)  # 规范字体格式

# 导入imageNet预训练模型
from torchvision.models import resnet18

# model = resnet18(pretrained=True).eval().to(device)  # pretrained代表是否载入官方已经训练好的权重
model = resnet34(num_classes=11)
model_weight_path = './ResNet_method.pth'
model.load_state_dict(torch.load(model_weight_path))
model = model.eval().to(device)
# 导入可解释性分析方法
from torchcam.methods import SmoothGradCAMpp, CAM

cam_extractor = CAM(model)

# 预处理
from torchvision import transforms
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 图像分类
img_path='img_1.png'
img_pil=Image.open(img_path)  # 以pil格式打开图片
input_tensor=test_transform(img_pil).unsqueeze(0).to(device)  # 图像处理

pred_logits=model(input_tensor)
pred_top1=torch.topk(pred_logits,1)
pred_id=pred_top1[1].detach().cpu().numpy().squeeze().item()  # 预测类别
print(pred_id)
print(pred_logits)


# 生成可解释性分析热力图
activation_map=cam_extractor(pred_id,pred_logits)
# print(activation_map,activation_map.shape)
activation_map=activation_map[0][0].detach().cpu().numpy()
print(activation_map)

plt.imshow(activation_map)
plt.show()

# 将activation_map覆盖在原图上
from torchcam.utils import overlay_mask
result=overlay_mask(img_pil,Image.fromarray(activation_map),alpha=0.7)
print(result)
plt.imshow(result)
plt.show()



