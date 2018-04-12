import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    """
    损失函数
    :param X:(np.matrix) 样本特征矩阵
    :param y:(np.matrix) 目标变量一维矩阵
    :param theta: (np.matrix)模型参数一维矩阵
    :return: 损失值
    """
    inner = np.power(((X * theta.T) - y), 2)  # 计算该模型下每个样本与实际值的误差
    return np.sum(inner) / (2 * len(X))  # 计算计算顺势函数

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))  # 存储每次迭代的theta
    slope_temp = np.matrix(np.zeros(theta.shape)) # 存储每次迭代的斜率slope
    parameters = int(theta.ravel().shape[1])  # theta的长度,用于计算theta的偏分
    cost = np.zeros(iters)  # 存储每次迭代的损失函数
    theta_list =[] # 存储每次迭代的theta
    slope_list =[]

    for i in range(iters):  # 循环计算每一次迭代
        error = (X * theta.T) - y  # (h_θ(X)-y)

        for j in range(parameters): # 循环计算每一个theta
            term = np.multiply(error, X[:, j])  # error*X_j
            slope = (1 / len(X)) * np.sum(term)  # 1/m * \sum(term) 加和求平均
            slope_temp[0,j] = slope
            gradient=alpha * slope  # α*slope  学习速率乘以斜率
            temp[0, j] = theta[0, j] - gradient  # 每次更新

        theta = temp  # 更新theta
        cost[i] = computeCost(X, y, theta)  #计算本次迭代的损失函数
        slope_list.extend(slope_temp.tolist())
        theta_list.extend(theta.tolist())

    return theta_list, slope_list, cost

def simp(a,b,c):  # 生成医院二次函数
    def equ(theta):  # 一元二次函数
        return a*theta*theta +b * theta +c
    def der(theta):  # 切线
        return 2*a*theta+b
    opti = -b/2*a  # 斜率
    return equ,der,opti

def coef(X,y):  # 根据特征矩阵和目标变量生成costfunction的化简形式
    m = len(X)
    a= np.power(X,2).sum()/(2*m)
    b= -np.sum(X.A*y.A*2)/(2*m)
    c= np.power(y,2).sum()/(2*m)
    return simp(a=a,b=b,c=c)

X=np.mat([[1],[3],[3]])  # 初始化数据
y=np.mat([[2],[1],[3]])
theta=np.mat([100])  # 初始化theta
alpha = 0.01
iters = 1000

theta_list, slope, cost = gradientDescent(X, y, theta, alpha, iters)  #
theta_list = [i[0] for i in theta_list]
equ,der,opti = coef(X,y)

x = np.linspace(opti-100,opti+100,1000)
y = [equ(i) for i in x]  # 生成costfun曲线

plt.plot(x,y)
plt.scatter(theta_list, cost)
plt.show()

import json
dic = [{"x1":i,"y1":equ(i),"x2":i+0.5,"y2":equ(i+0.5)} for i in x]  # 生成cost曲线
json.dump(dic,open("./test.json","w"))
dic1 = [{"cx":i,"cy":j} for i,j in zip(theta_list, cost)] # 生成每次迭代的结果
json.dump(dic1,open("./test1.json","w"))


# 画拟合曲线
g, slope, cost = gradientDescent(X, y, theta, alpha, iters)
x = np.linspace(0,5,10)
f = (g[-1][0] * x)  # 预测值

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter([1,3,3], [2,1,3], label='Traning Data')
ax.legend(loc=2)
plt.show()

# 画拟合曲线
X=np.mat([[1,1],[1,3],[1,3]])
y=np.mat([[2],[1],[3]])
theta=np.mat([0,0])  # 初始化theta
alpha = 0.01
iters = 1000

g, slope, cost = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(0,5,10)
f = g[-1][0] + (g[-1][1] * x)  # 预测值

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter([1,3,3], [2,1,3], label='Traning Data')
ax.legend(loc=2)
plt.show()
