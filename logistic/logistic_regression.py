import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def computerCost(theta, X, y):
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))  # 存储每次迭代的theta
    slope_temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)  # 存储每次迭代的损失函数
    theta_list =[] # 存储每次迭代的theta
    slope_list =[]

    for i in range(iters):
        error = sigmoid(X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            slope = (1 / len(X)) * np.sum(term)  # 1/m * \sum(term) 加和求平均
            slope_temp[0,j] = slope
            gradient = alpha * slope
            temp[0, j] = theta[0, j] - gradient

        theta = temp  # 更新theta
        cost[i] = computerCost(X, y, theta)  # 计算本次迭代的损失函数
        slope_list.extend(slope_temp.tolist())
        theta_list.extend(theta.tolist())

    return theta_list, slope_list, cost


# 没有办法
X=np.mat([[1],[2]])  # 初始化数据
y=np.mat([[0],[1]])
theta=np.mat([100])  # 初始化theta
alpha = 0.01
iters = 100

theta_list, slope, cost = gradientDescent(X, y, theta, alpha, iters)  #
theta_list = [i[0] for i in theta_list]

plt.scatter(theta_list, cost)
plt.show()



def logistic(theta):
    def sig(X):
        z = (X*theta.T)
        h = 1 / (1 + np.exp(-z))
        return z, h
    return sig

X = np.mat([[0,1,2],[0,2,1]])
theta = np.mat([[1,1,1]])

logist=logistic(theta)
z,h = logist(X)



# 画sigmoid函数


import json
dic = [{"x":i/10,"y":sigmoid(i/10)} for i in range(-100,100)]
json.dump(dic,open("./test3.json","w"))


