import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()  # 导入数据集

iris.keys()  # 查看数据集的构成

print(iris['DESCR'])  # 查看数据集的描述

iris['feature_names']  # 查看自变量的名称

iris['target_names']  # 查看因变量的名称

iris['target']  # 查看因变量数据

iris['data']  # 查看自变量数据

X = iris['data']  # 获取自变量数据

y = iris['target']  # 获取因变量数据

target_names = iris['target_names']  # 获取因变量名称


# 观察数据集 展示图形，这里我们只使用sepal length和sepal width两个属性。
for m,i,target_names in zip('vo^',range(2),target_names[0:2]):
    sl = X[y == i,0]  # sl = sepal length (cm)
    sw = X[y == i,1]  # sw = sepal width (cm)
    plt.scatter(sl,sw,marker=m,label=target_names,s=30,c='k')

plt.xlabel('sepal length (cm)')  # 绘制x轴和y轴标签名
plt.ylabel('sepal width (cm)')

# 压缩后图形
X=X[(y==1) | (y==0),0:2]  # 获取sepal length和sepal width两个属性的自变量矩阵
y=y[(y==1) | (y==0)]  # 获取sepal length和sepal width两个属性的因变量矩阵

lda = LinearDiscriminantAnalysis(n_components=1)  # 创建模型变量，并设置压缩之后的维度

ld = lda.fit(X,y)  # 训练数据

X_t =ld.transform(X)  # 将模型应用到原矩阵上，降维

y_t = np.zeros(X_t.shape)  # 因为压缩到1维所以y轴坐标全部为0

for m,i,target_names in zip('ov^',range(3),target_names):  # 做压缩后的图像
    plt.scatter(X_t[y == i],y_t[y == i],marker=m,label=target_names,s=30,c='k')
plt.legend()
plt.grid()

# 分类
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)  # 分割训练集和测试集

lda = LinearDiscriminantAnalysis(n_components=1)  # 创建线性判别对象

ld = lda.fit(X_train,y_train)  # 训练模型

pre = ld.predict(X_test)  # 模型预测

list(zip(pre,y_test,pre==y_test))  # 查查看预测结果

ld.score(X_test,y_test) # 查看正确率


