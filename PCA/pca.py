import numpy as np
a=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])

a=a.T

c=1/len(a)*a.T.dot(a) # 协方差矩阵
eig=np.linalg.eig(c)
c=eig[1][:,0]  # 求第一个特征向量

touying1=a.dot(c.T)
li=[]
for i in range(len(touying1)):
    li.append(touying1[i]*a[i])
li=np.array(li)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(a)
X_new = pca.transform(a)


test = np.array([1,2])
test.repeat(2,axis=1)