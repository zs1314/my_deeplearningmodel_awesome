"""
demo
"""
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchcam.utils import overlay_mask

from model import *
import time
import torch
import torchvision
from PIL import Image
from model import *
import cv2
from PIL import ImageFont, ImageDraw
from torchcam.methods import CAM  # 导入可解释性方法
from flask import Flask, request

# 由于opencv没办法写入中文,这里借助PIL形式写入汉字
font = ImageFont.truetype('SimHei.ttf', 32)
font_img = ImageFont.truetype('SimHei.ttf', 10)

# 注意:opencv是不能写中文的，所以只能用PIL
img_path = './img_video/test_img.jpg'  # 图片保存路径
video_path = './img_video/test_video.mp4'  # 视频保存路径
# 标签（11个类别）
labels = ['欧塞瓦', '品丽珠', '赤霞珠', '霞多丽', '梅洛', '米勒图高', '黑皮诺', '雷司令', '长相思 ', '西拉', '丹魄']


def my_cam(img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载模型
    model = resnet34(num_classes=11)

    # 导入可解释性分析方法
    from torchcam.methods import SmoothGradCAMpp

    cam_extractor = SmoothGradCAMpp(model)

    # 加载权重数据
    model_weight_path = './ResNet_method.pth'
    model.load_state_dict(torch.load(model_weight_path))
    model = model.eval().to(device)

    # 预处理
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    # 图像分类
    img_pil = Image.fromarray(img)  # 以pil格式打开图片
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)  # 图像处理
    pred_logits = model(input_tensor)
    pred_top1 = torch.topk(pred_logits, 1)
    pred_id = pred_top1[1].cpu().detach().numpy().squeeze().item()  # 预测类别
    print(pred_id)
    print(pred_logits)

    # 前5的概率
    predict = torch.softmax(pred_logits, dim=1)
    top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
    pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

    # 生成可解释性分析热力图
    activation_map = cam_extractor(pred_id, pred_logits)
    # activation_map=activation_map.to('cpu')
    # print(activation_map.is_cuda)
    # print(activation_map,activation_map.shape)
    activation_map = activation_map[0][0].cpu().detach().numpy()
    print(activation_map)

    # plt.imshow(activation_map)
    # plt.show()

    # 在图像上写字（PIL）
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(confs)):
        pred_class = labels[pred_idx[i]]  # 类别
        text = str(pred_class) + str('  ') + str(confs[i])  # 打印出来的格式
        print(text)
        # 将文字打印在图片上
        # 文字坐标、中文字符串、字体、颜色（pil）
        draw.text((50, 100 + 50 * i), text, font=font, fill=(116, 255, 144))

    #     两种形式都行
    print("预测结果为：", labels[pred_idx[0]])
    print("概率为：", confs[0])  # item()可以直接显示数值

    # 将activation_map覆盖在原图上
    from torchcam.utils import overlay_mask
    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
    print(result)
    # plt.imshow(result)
    # plt.show()
    result = np.array(result)
    return result


# 自定义处理图像函数,也可以不处理直接返回
def process_frame(img):
    # 记录该帧开始的时间
    srart_time = time.time()

    # 引入图片

    # 图片转换
    img = Image.fromarray(img)  # 将图片转换为PIL形式,方便转换为tensor数据类型
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

        # 在图像上写字（PIL）
        draw = ImageDraw.Draw(frame)
        for i in range(len(confs)):
            pred_class = labels[pred_idx[i]]  # 类别
            text = str(pred_class) + str('  ') + str(confs[i])  # 打印出来的格式
            print(text)
            # 将文字打印在图片上
            # 文字坐标、中文字符串、字体、颜色（pil）
            draw.text((50, 100 + 50 * i), text, font=font, fill=(116, 255, 144))

        #     两种形式都行
        print("预测结果为：", labels[output.argmax(1)], labels[pred_idx[0]])
        print("概率为：", predict[0][output.argmax(1)].numpy().item(), confs[0])  # item()可以直接显示数值

    # 记录该帧处理完毕的时间
    end_time = time.time()

    # 计算每秒处理图像帧数
    FPS = 1 / (end_time - srart_time)

    # 打印数据
    draw.text((50, 50), 'FPS' + '  ' + str(int(FPS)), font=font, fill=(116, 255, 144))

    frame = np.array(frame)  # PIL准尉array(opencv打开需要的是array类型)
    return frame


def predict_img(img_path):
    # 自定义处理图像函数,也可以不处理直接返回

    # 引入图片
    img = Image.open(img_path)  # pil格式打开
    # 图片转换
    img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）
    frame = img  # 定义一个等会处理完后返回的
    # 图片转换
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                 torchvision.transforms.ToTensor()]
                                                )
    img = transforms(img)
    img = torch.reshape(img, (1, 3, 32, 32))  # 转换为适合进入神经网络的类型格式，添加一个batchsize维度
    img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

    # 标签（11个类别）
    labels = ['欧塞瓦', '品丽珠', '赤霞珠', '霞多丽', '梅洛', '米勒图高', '黑皮诺', '雷司令', '长相思 ', '西拉', '丹魄']

    # 建立模型
    my_ResNet = resnet34(num_classes=11)

    # 加载权重数据
    model_weight_path = './ResNet_method.pth'
    my_ResNet.load_state_dict(torch.load(model_weight_path))
    my_ResNet = my_ResNet.eval().to('cuda')

    # 开始预测
    my_ResNet.eval()
    with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
        output = my_ResNet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
        output = output.to('cpu')
        print(output)
        predict = torch.softmax(output, dim=1)
        # print(predict)
        predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）

        top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
        pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

        print(pred_idx)
        print(confs)

        # # 生成可解释性分析热力图（耗时部分）
        # pred_top1 = torch.topk(predict, 1)
        # pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()  # 预测类别
        # activation_map = cam_extractor(pred_id, output)
        # activation_map = activation_map[0][0].detach().cpu().numpy()
        # plt.imshow(activation_map)
        # plt.show()
        # frame = overlay_mask(frame, Image.fromarray(activation_map), alpha=0.7)
        #

        # 在图像上写字
        draw = ImageDraw.Draw(frame)
        for i in range(len(confs)):
            pred_class = labels[pred_idx[i]]  # 类别
            text = str(pred_class) + str('  ') + str(confs[i])  # 打印出来的格式
            print(text)
            # 将文字打印在图片上
            # 文字坐标、中文字符串、字体、颜色（pil）
            draw.text((0, 20 + 20 * i), text, font=font_img, fill=(116, 255, 144))

        #     两种形式都行
        print("预测结果为：", labels[output.argmax(1)], labels[pred_idx[0]])
        print("概率为：", predict[0][output.argmax(1)].numpy().item(), confs[0])  # item()可以直接显示数值
        frame.show()
    return frame


# 调用摄像头拍摄照片并保存
def get_img():
    # 延迟2秒
    time.sleep(2)
    # 调用摄像头,0是默认摄像头，1是外置摄像头
    cap = cv2.VideoCapture(0)
    # 捕获并处理一帧画面
    success, frame = cap.read()
    if not success:
        print('error!')
    frame = process_frame(frame)
    # 关闭摄像头
    cap.release()
    # 关闭窗口
    cv2.destroyAllWindows()
    # 保存图片
    cv2.imwrite(img_path, frame)
    print('图片已保存', img_path)


# 调用摄像头捕获并保存视频
def get_video():
    # 调用摄像头,0是默认摄像头，1是外置摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 返回值就是视屏数据
    # 打开cap
    cap.open(0)
    # 视频尺寸
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 文件编码方式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(video_path, fourcc, fps,
                          (int(frame_size[0]), int(frame_size[1])))

    # --!!!关键部分，无限循环--
    while cap.isOpened():  # 判断摄像头是否成功打开
        # 获取画面
        success, frame = cap.read()  # read()读取一帧画面,返回值两个参数:是否成功(bool),那一帧画面
        # 未捕获到，报错并退出
        if not success:
            print('error!')
            break
        # 对每帧进行处理
        # frame = process_frame(frame)

        # 热力图
        frame = my_cam(frame)
        # 将处理后的帧写入视频文件
        out.write(frame)

        # 实时显示图像
        cv2.imshow('press q to quit', frame)  # 第一个参数为窗口名称
        # 按q或esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # 关闭图像窗口
    cv2.destroyAllWindows()
    out.release()
    # 关闭摄像头
    cap.release()  # 释放资源
    print('视频已保存', video_path)





def main():
    # get_img()
    # get_video()
    predict_img(r'D:\pythonProject1\ResNet_ui_nn\img_1.png')
    # my_cam(r'D:\pythonProject1\ResNet_ui_nn\img_1.png')
    # predict_flask()


if __name__ == '__main__':
    main()
