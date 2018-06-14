import numpy as np
a=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])
a=a.T
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(a)

X_new = pca.transform(a)

u,s,v = np.linalg.svd(a)
s=s.tolist()
s.extend([0,0,0])


temp1=u.dot(np.diag(s))
temp1=temp1[:,0:2]
temp1.dot(v)