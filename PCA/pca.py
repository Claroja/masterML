import numpy as np
a=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])

a=a.T

c=1/len(a)*a.T.dot(a) # 协方差矩阵
eig=np.linalg.eig(c)
c=eig[1][:,0]  # 求第一个特征向量

touying1=a.dot(c.T)
li=c.reshape(1,len(c)).repeat(len(touying1),axis=0)
touying1=touying1.reshape(len(touying1),1).repeat(2,axis=1)
touying=li*touying1

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(a.T[0],a.T[1],c='red')
plt.scatter(touying.T[0],touying.T[1])
plt.plot([0,0.707],[0,0.707])
plt.xlim([-10,10])
plt.ylim([-10,10])


from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(a)
X_new = pca.transform(a)


test = np.array([1,2])
test.repeat(2,axis=1)