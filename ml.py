# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


class AdalineGd(object):
    '''
    eta: float
    学习效率，处于0和1之间
    n_iter：int
    对训练数据进行学习，改进次数
    w_：一维向量
    存储权重数值
    error_：
    一维向量
    存储每次迭代改进时，神经网络对数据进行错误判断的次数
    '''
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''
        :param X: 二维数组[n_samples, n_features]
        n_samples 表示X中含有训练数据条目数
        n_features含有4个数据的一维向量，用于表示一条训练条目
        :param y: 一维向量
        用于存储每一训练条目对应的正确分类
        :return:
        '''
        self.w_ = np.zeros(1+X.shape[1])  # 权重初始化为零
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)  # 向量
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]+self.w_[0])

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)


a = 'https://archive.ics.uci.edu/ml'
b = '/machine-learning-databases/iris/iris.data'
file = (a + b)
df = pd.read_csv(file, header=None)
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
# 根据整数位置选取单列或单行数据
X = df.loc[0:100, [0, 2]].values


def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


ada = AdalineGd(eta=0.0001, n_iter=100)
ada.fit(X, y)
plot_decision_region(X, y, classifier=ada)
plt.xlabel('huajingchang')
plt.ylabel('huabanchang')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('epochs')
plt.ylabel('sum-squard-error')
plt.show()
