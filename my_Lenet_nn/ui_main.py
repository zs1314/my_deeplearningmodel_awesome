"""
导入模块和库
"""
from PyQt5.uic.properties import QtGui
from ui1_window import Ui_Form
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from demo import *

class my_Window(QWidget, Ui_Form):
    # 初始化
    def __init__(self):
        super(my_Window, self).__init__()
        self.pre = None
        self.directory1 = None
        self.setupUi(self)

        # 按钮与方法建立联系
        self.pushButton_jiance.clicked.connect(self.jiance)
        self.pushButton_model.clicked.connect(self.select_model)
        self.pushButton_image.clicked.connect(self.openImage)

    # 导入图片
    def openImage(self):
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量
        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是默认打开的路径，第四个参数是需要的格式
        imgNamepath, imgType = QFileDialog.getOpenFileName(self, "选择图片",
                                                           "",
                                                           "*.jpg;;*.png;;All Files(*)")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        my_img = QPixmap(imgNamepath).scaled(self.image.width(), self.image.height())
        print(imgNamepath)
        # 在label控件上显示选择的图片
        self.image.setPixmap(my_img)

    # 检测图片
    def jiance(self):
        self.pre=predict()
        self.textBrowser.setText(self.pre.my_predict(imgNamepath,self.directory1[0]))

    # 选择模型（这里具体应该为选择模型的参数）
    def select_model(self):
        # 其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        self.directory1 = QFileDialog.getOpenFileName(self,'选择模型')
        print(self.directory1[0])


# 主程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = my_Window()
    window.show()
    sys.exit(app.exec_()) # 不间断的运行程序
