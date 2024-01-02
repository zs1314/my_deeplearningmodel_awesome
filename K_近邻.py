# -*- coding: utf-8 -*-
"""
利用sklearn封装好的函数实现Knn算法
"""

# 调用相关的库和模块
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class KNN(object):
    # 获取鸢尾花数据 三个类别(山鸢尾/0，虹膜锦葵/1，变色鸢尾/2)，每个类别50个样本，每个样本四个特征值(萼片长度，萼片宽度，花瓣长度，花瓣宽度)
    def get_iris_data(self):
        iris = load_iris()
        iris_data = iris.data  # 特征值
        iris_target = iris.target  # 目标值（标签、类别）
        print(iris_data)
        print(iris_target)
        return iris_data, iris_target

    def run(self):
        # 1.调用定义的静态方法，获取鸢尾花的特征值，目标值
        iris_data, iris_target = self.get_iris_data()

        # 2.将数据分割成训练集和测试集 ，train_test_split中X，y参数为要分割的数据，X为特征值数据，y为目标值数据
        # test_size=0.25表示将25%的数据用作测试集
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)

        # 3.特征工程(对特征值进行标准化处理)
        std = StandardScaler()
        # 对训练集的特征值标准化
        x_train = std.fit_transform(x_train)
        # 对测试集的特征值标准化
        x_test = std.transform(x_test)
        # 注意：鸢尾花数据集中目标值用0,1,2表示，故不需要标准化来减小数据对模型的影响

        # 4、运用交叉验证法，选取准确率最高的超参数，不用人工调参
        scores = []
        ks = []
        for k in range(1, 50):
            knn = KNeighborsClassifier(n_neighbors=k)
            cross_score = cross_val_score(knn, x_train, y_train, cv=6).mean()  # 计算超参数为k时的准确率
            scores.append(cross_score)
            ks.append(k)

        # 将k和对应的score转换为narry形式
        ks_arr = np.array(ks)
        scores_arr = np.array(scores)

        # 将学习曲线可视化
        plt.plot(ks_arr, scores_arr)  # 传入x,y
        plt.xlabel('K')  # 对X轴命名
        plt.ylabel('Score')  # 对Y轴命名
        plt.show()  # 展示学习曲线

        # 取最大值的下标
        max_idx = scores_arr.argmax()
        # 最大值对应的k值
        max_k = ks[max_idx]
        print('最大值下标：', max_idx)
        print('最大值对应的k值：', max_k)

        # 5、将K值传入算法
        knn = KNeighborsClassifier(max_k)

        # 开始拟合模型（调用fit(X,y),X为特征值，y为目标值），x_train作为特征值，y_train作为标签（目标值）
        knn.fit(x_train, y_train)

        # 获取预测结果，predict()方法的返回值为待测试样例的目标值的数组（这里一维张量，即向量）
        y_predict = knn.predict(x_test)

        # 获取测试集中每个样本通过Knn算法后，得到的对每个类别的各个概率，predict_proba的返回值是多维列表
        probil = knn.predict_proba(x_test)

        # 预测结果展示
        labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]  # 由于该数据集target用0,1,2三个数字来表示类别，故提前写好标签
        for i in range(len(y_predict)):
            print("第%d次测试:真实值:%s\t预测值:%s\t预测概率：%s" % (
                (i + 1), labels[y_test[i]], labels[y_predict[i]], probil[i]))

        # knn.score是得到X相对于y的平均准确度，X为待测试案例，y为测试的真实目标值
        print("准确率：", knn.score(x_test, y_test))


# 主程序
if __name__ == '__main__':
    knn = KNN()  # 调用类，创建一个实例对象
    knn.run()  # 调用类中核心（run）方法

