# __author__ = 'Xiang'
# coding:utf-8
from matplotlib import pylab as plt
import numpy as np


def read_file(filename):
    f = open(filename, 'r')
    d = f.readlines()
    f.close()
    return d


class PCA(object):
    """
    使用PCA对高维数据进行降维处理
    """
    def __init__(self):
        data = read_file('data.txt')

        self.n = len(data)  # 数据的个数
        self.dim = 4  # 原始数据的维度
        self.x = np.zeros((self.n, self.dim), dtype='float64')
        for i in range(self.n):
            data_ = data[i].split(',')
            self.x[i][0] = data_[0]
            self.x[i][1] = data_[1]
            self.x[i][2] = data_[2]
            self.x[i][3] = data_[3]

        self.k = 2  # 降到二维
        self.mean_x = np.zeros((self.n, self.dim), dtype='float64')  # 原始数据减去均值以后的x
        self.mean = np.zeros((self.dim, 1), dtype='float64')  # x的均值
        self.cov = np.zeros((self.dim, self.dim), dtype='float64')  # 协方差矩阵
        self.pre_x = np.zeros((self.n, self.dim), dtype='float64')  # 预处理之后的数据
        self.eig_val = np.zeros((1, self.dim), dtype='float64')  # 特征值
        self.eig_vec = np.zeros((self.dim, self.dim), dtype='float64')  # 特征向量
        self.final_x = np.zeros((self.n, self.k), dtype='float64')  # 投影后的数据
        self.pretreatment()
        self.pca()

    def pretreatment(self):
        """
        预处理
        """
        # 求均值
        for i in range(self.dim):
            self.mean[i] = np.mean(self.x[:, i])

        for i in range(self.n):
            self.mean_x[i] = self.x[i] - self.mean.T

        # 求协方差
        # self.cov = np.cov(self.mean_x, rowvar=0)
        # mean_x已经是x减去均值了，所以直接相乘就是方差
        self.cov = self.mean_x.T.dot(self.mean_x) / self.n

        for i in range(self.dim):
            self.pre_x[:, i] = self.mean_x[:, i] / np.sqrt(self.cov[i][i])   # x的每个维度都处理一次

    def pca(self):
        """
        pca的实现
        """
        # 需要注意的是，在这里需要对预处理之后的数据重新计算协方差
        # 计算均值
        for i in range(self.dim):
            self.mean[i] = np.mean(self.pre_x[:, i])

        # 计算协方差
        for i in range(self.n):
            self.mean_x[i] = self.pre_x[i] - self.mean.T
        self.cov = (self.mean_x.T.dot(self.mean_x)) / self.n

        # 求特征值
        self.eig_val, self.eig_vec = np.linalg.eig(np.mat(self.cov))

        # eig_vec的列向量是特征向量，所以用eig_v存储特征向量，以便后面排序
        eig_v = np.zeros((self.dim, self.dim), dtype='float64')
        for i in range(self.dim):
            eig_v[i] = self.eig_vec[:, i].T

        # 排序
        eig_list = zip(self.eig_val, eig_v)
        eig_list.sort(key=lambda g: g[0], reverse=True)

        # 处理排序的结果
        for i in range(len(eig_list)):
            self.eig_val[i] = eig_list[i][0]
            self.eig_vec[i] = eig_list[i][1]

        # 最大的k个特征向量，降维
        f_vec = self.eig_vec[0:self.k, :]
        self.final_x = self.pre_x * f_vec.T

        # 显示降维后的数据点
        plt.plot(self.final_x[0:50, 0], self.final_x[0:50, 1], 'bo')
        plt.plot(self.final_x[50:100, 0], self.final_x[50:100, 1], 'go')
        plt.plot(self.final_x[100:150, 0], self.final_x[100:150, 1], 'ro')
        plt.show()


if __name__ == '__main__':
    a = PCA()
