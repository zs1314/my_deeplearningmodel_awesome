"""
demo
"""
import time

import torch
import torchvision
from PIL import Image
from model import *
import cv2

img_path = './img_video/test_img.jpg'  # 图片保存路径
video_path = './img_video/test_video.mp4'  # 视频保存路径


# 自定义处理图像函数,也可以不处理直接返回
def process_frame(img):
    # 引入图片
    # img = Image.open(img_path)  # 将图片转换为PIL形式,方便转换为tensor数据类型


    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    frame=Image.fromarray(frame)

    # 图片转换
    img=Image.fromarray(img)
    img = img.convert('RGB')  # 以此来适应各种格式图片（png,jpg）

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                  (0.229, 0.224, 0.225))
                                                 ])

    img = transforms(img)
    img = torch.reshape(img, (1, 3, 224, 224))  # 转换为适合进入神经网络的类型格式
    img = img.cuda()  # 以CUDA设备来demo，因为神经网络模型参数等也为cuda训练，传入验证图片时也必须在cuda上

    # 标签（三个类别）
    labels = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']

    # 建立模型
    my_ResNet = resnet34(num_classes=5)
    # 加载权重数据
    model_weight_path = './ResNet_method.pth'
    my_ResNet.load_state_dict(torch.load(model_weight_path))
    my_ResNet = my_ResNet.to('cuda')
    # 开始预测
    my_ResNet.eval()
    with torch.no_grad():  # 保证模型参数数据不受影响（不需要用到梯度）
        output = my_ResNet(img)  # output此时为一维张量，存放着各个类别的非线性概率（加和不为1）
        predict = torch.softmax(output, dim=0)
        predict = predict.to('cpu')  # 注意：这里先要转换为cpu上运转，便于tensor数据类型转换为numpy（！！！debug一下午才找到，惨痛教训）
        print("预测结果为：", labels[output.argmax(1)])
        print("概率为：", predict[0][output.argmax(1)].item())  # item()可以直接显示数值

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

# def predict_img():
#    # 调用摄像头捕获并保存视频
def get_video():
    # 调用摄像头,0是默认摄像头，1是外置摄像头
    cap =cv2.VideoCapture(0,cv2.CAP_DSHOW)
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
        process_frame(frame)

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


if __name__ == '__main__':
    main()


