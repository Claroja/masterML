from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
print(boston['DESCR'])
boston['target']
boston['data'].head()



from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()  # 建立线性模型对象
boston = datasets.load_boston()  # 获得波士顿数据集
y = boston['target'] # 获得波士顿目标变量


predicted = cross_val_predict(lr, boston.data, y, cv=10)  # 交叉验证

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
