import numpy as np
from numpy.linalg import inv,eig
import matplotlib.pyplot as plt
# x1 = np.array([[4,2],[2,4],[2,3],[3,6],[4,4]])
# x2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

def lda(x1,x2):
    mu1 = x1.mean(axis=0)
    mu2 = x2.mean(axis=0)
    s1 = np.cov(x1.T)  # 计算协方差矩阵
    s2 = np.cov(x2.T)
    sw = s1 + s2
    sb = ((mu1-mu2).reshape(2,1))*(mu1-mu2)  # 数组只有一行的时候不能直接使用转置T，而需要使用reshape
    si = inv(sw).dot(sb)  # 计算逆矩阵
    return eig(si)  # 计算特征值和特征向量,返回的特征向量，是竖着看，第一行是y的系数，第二行是x的系数

