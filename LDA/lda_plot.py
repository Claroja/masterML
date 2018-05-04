feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}
import pandas as pd

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

df.tail()



from sklearn.preprocessing import LabelEncoder

X = df.iloc[:,0:4].values
y = df['class label'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

from matplotlib import pyplot as plt
import numpy as np
import math


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))


# for ax,cnt in zip(axes.ravel(), range(4)):

ax = axes.ravel()[0]
cnt = 0

min_b = math.floor(np.min(X[:,cnt]))  # 获得该维度的最小值
max_b = math.ceil(np.max(X[:,cnt]))  # 获得该维度的最大值
bins = np.linspace(min_b, max_b, 25)  # 获得柱状图的x轴坐标

for lab, col in zip(range(1, 4), ('blue', 'red', 'green')): # 做柱状图
    ax.hist(X[y == lab, cnt],  # 获得标签为当前标签值的样本，并取第0列.相当于[列表,值]的索引方法
            color=col,  # 设置为当前颜色
            label='class %s' % label_dict[lab],
            bins=bins,
            alpha=0.5, )

ylims = ax.get_ylim()


leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
leg.get_frame().set_alpha(0.5)  # 设置图例边框的透明度
ax.set_ylim([0, max(ylims)+2])  # y坐标轴加2
ax.set_xlabel(feature_dict[cnt])  # 设置x轴的标签
ax.set_title('Iris histogram #%s' %str(cnt+1))  # 设置标题

ax.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")  # 去掉ticks

ax.spines["top"].set_visible(False)  # 去掉坐标轴
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)


axes[0][0].set_ylabel('count')  # 设置0行0列的y坐标标签
axes[1][0].set_ylabel('count')  # 设置1行0列的y坐标标签

fig.tight_layout()  # 自动调整图像


 # 第一步求，每一个分类的平均值
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))



# 计算协方差矩阵
S_W = np.zeros((4,4))
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)