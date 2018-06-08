import numpy as np
a=np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])
a=a.T
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(a)

X_new = pca.transform(a)

np.linalg.svd(a.T)