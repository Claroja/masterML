import numpy as np
a=np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
u,s,v = np.linalg.svd(a)