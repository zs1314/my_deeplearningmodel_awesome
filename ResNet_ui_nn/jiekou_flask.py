# 图片转换
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


def predict(img):
    # img = Image.fromarray(img)  # 将图片转换为PIL形式,方便转换为tensor数据类型
    frame = img  # 定义一个等会处理完后返回的
    img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）

    # 图片转换
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                 torchvision.transforms.ToTensor()]
                                                )
    img = transforms(img)
    img = torch.reshape(img, (1, 3, 32, 32))  # 转换为适合进入神经网络的类型格式
    img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

    # 标签（11个类别）
    labels = ['欧塞瓦', '品丽珠', '赤霞珠', '霞多丽', '梅洛', '米勒图高', '黑皮诺', '雷司令', '长相思 ', '西拉', '丹魄']

    # 建立模型
    my_ResNet = resnet34(num_classes=11)

    # 加载权重数据
    model_weight_path = './ResNet_method.pth'
    my_ResNet.load_state_dict(torch.load(model_weight_path))
    my_ResNet = my_ResNet.eval().to('cuda')  # 这个eval一定要加，主要是为了关闭梯度，为热力图做准备
    # for param in my_ResNet.parameters():
    #     print(param.requires_grad)

    # 开始预测
    my_ResNet.eval()
    with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
        output = my_ResNet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
        output.to('cpu')
        predict = torch.softmax(output, dim=1)
        predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）

        top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
        pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

        print(pred_idx)
        print(confs)

        #     两种形式都行
        print("预测结果为：", labels[output.argmax(1)], labels[pred_idx[0]])
        print("概率为：", predict[0][output.argmax(1)].numpy().item(), confs[0])  # item()可以直接显示数值

    frame = np.array(frame)  # PIL准尉array(opencv打开需要的是array类型)
    template = "class:{:<15} proability:{:.3f}"
    # text = [template.format(labels[pred_idx[0]], confs[0])]
    img_pre = [(labels[pred_idx[0]], confs[0]), (labels[pred_idx[1]], confs[1]), (labels[pred_idx[2]], confs[2]),
               (labels[pred_idx[3]], confs[3]), (labels[pred_idx[4]], confs[4])]
    text = [template.format(k, v) for k, v in img_pre]
    print(text)
    return_info = {"result": text}
    return return_info


@app.route('/', methods=["POST"])
@cross_origin()
def predict_flask():
    # uniapp端
    # image = request.files['file']
    # print(image)
    # img = image.read()
    # print(img)
    # img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    # print(img)
    # print('000')
    # return jsonify(predict(img))

    # 小程序端
    img_base64 = request.form.get('picture')
    image = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(image))
    print(image)
    return jsonify(predict(image))


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
#host='0.0.0.0'
