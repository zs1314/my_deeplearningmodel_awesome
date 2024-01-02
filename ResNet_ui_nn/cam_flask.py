import base64
import io

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from flask import Flask, request, jsonify

from ResNet_ui_nn.model import resnet34
from flask_cors import *

app = Flask(__name__)


def cam(img_pil):
    labels = ['欧塞瓦', '品丽珠', '赤霞珠', '霞多丽', '梅洛', '米勒图高', '黑皮诺', '雷司令', '长相思 ', '西拉', '丹魄']
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch

    from ResNet_ui_nn.model import resnet34

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 导入pillow的中文字体（opencv没有中文）
    from PIL import ImageFont, ImageDraw

    # font = ImageFont.truetype('SimHei.ttf', 50)  # 规范字体格式

    model = resnet34(num_classes=11)
    model_weight_path = './ResNet_method.pth'
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval().to(device)
    # 导入可解释性分析方法
    from torchcam.methods import CAM

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
    # img_pil = Image.open(img_path)  # 以pil格式打开图片
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)  # 图像处理

    pred_logits = model(input_tensor)
    pred_top1 = torch.topk(pred_logits, 1)
    pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()  # 预测类别
    print(pred_id)
    print(pred_logits)

    predict = torch.softmax(pred_logits, dim=1)
    predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）

    top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
    pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

    # 生成可解释性分析热力图
    activation_map = cam_extractor(pred_id, pred_logits)
    # print(activation_map,activation_map.shape)
    activation_map = activation_map[0][0].detach().cpu().numpy()
    # print(activation_map)

    # plt.imshow(activation_map)
    # plt.show()

    # 将activation_map覆盖在原图上
    from torchcam.utils import overlay_mask
    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
    print(result)
    # plt.imshow(result)
    # plt.show()
    # result = np.array(result)
    byte_data = io.BytesIO()  # 创建一个字节流管道
    result.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    pro=str(confs[0])
    result_info = {"result": base64_str, "class": labels[pred_id], "pro": pro}
    return result_info


@app.route('/', methods=["POST"])
@cross_origin()
def cam_flask():
    img_base64 = request.form.get('picture')
    image = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(image)).convert('RGB')
    print(image)
    # return jsonify(cam(image))
    return jsonify(cam(image))


if __name__ == "__main__":
    app.run(debug=True, port=4000,host='0.0.0.0')
