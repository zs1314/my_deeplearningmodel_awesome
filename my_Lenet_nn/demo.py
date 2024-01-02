"""
demo
"""
import numpy as np
import torch
import torchvision
from PIL import Image
from model import *
import time
import torch
import torchvision
from PIL import Image
from model import *
import cv2
from PIL import ImageFont, ImageDraw

# 由于opencv没办法写入中文,这里借助PIL形式写入汉字
font = ImageFont.truetype('SimHei.ttf', 32)
font_img = ImageFont.truetype('SimHei.ttf', 10)

# 注意:opencv是不能写中文的，所以只能用PIL
img_path = './img_video/test_img.jpg'  # 图片保存路径
video_path = './img_video/test_video.mp4'  # 视频保存路径


# 自定义处理图像函数,也可以不处理直接返回
def process_frame(img):
    # 记录该帧开始的时间
    srart_time = time.time()

    # 引入图片

    # 图片转换
    img = Image.fromarray(img)  # 将图片转换为PIL形式,方便转换为tensor数据类型
    img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）
    frame = img  # 定义一个等会处理完后返回的
    # 图片转换
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                 torchvision.transforms.ToTensor()]
                                                )
    img = transforms(img)
    img = torch.reshape(img, (1, 3, 32, 32))  # 转换为适合进入神经网络的类型格式
    img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

    # 标签（10个类别）
    labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '火车']

    # 建立模型,加载模型(训练得到的，选误差最小的)
    my_lenet = torch.load('./method/LeNet_method_64.pth')
    my_lenet(img)

    # 开始预测
    my_lenet.eval()
    with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
        output = my_lenet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
        predict = torch.softmax(output, dim=1)
        predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）

        top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
        pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

        print(pred_idx)
        print(confs)

        # 在图像上写字
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
    img = Image.open(img_path)
    # 图片转换
    img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）
    frame = img  # 定义一个等会处理完后返回的
    # 图片转换
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                 torchvision.transforms.ToTensor()]
                                                )
    img = transforms(img)
    img = torch.reshape(img, (1, 3, 32, 32))  # 转换为适合进入神经网络的类型格式
    img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

    # 标签（10个类别）
    labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '火车']

    # 建立模型,加载模型(训练得到的，选误差最小的)
    my_lenet = torch.load('./method/LeNet_method_64.pth')
    my_lenet(img)

    # 开始预测
    my_lenet.eval()
    with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
        output = my_lenet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
        predict = torch.softmax(output, dim=1)
        predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）

        top_n = torch.topk(predict, 5)  # 取置信度最大的5个结果
        pred_idx = top_n[1].cpu().detach().numpy().squeeze()  # 预测类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 前五的概率

        print(pred_idx)
        print(confs)

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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        # 未捕获到，报错并退出
        if not success:
            print('error!')
            break
        # 对每帧进行处理
        frame = process_frame(frame)

        # 将处理后的帧写入视频文件
        out.write(frame)

        # 实时显示图像
        cv2.imshow('press q to quit', frame)
        # 按q或esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # 关闭图像窗口
    cv2.destroyAllWindows()
    out.release()
    # 关闭摄像头
    cap.release()
    print('视频已保存', video_path)


def main():
    # get_img()
    get_video()
    # predict_img(r'D:\pythonProject1\my_Lenet_nn\imgs_predict\dog.jpg')


if __name__ == '__main__':
    main()
