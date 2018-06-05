import numpy as np
p=np.array([[1,0]])
m=np.array([[1,1]])

u_m=m/np.linalg.norm(m)

touying = p.dot(u_m.T)
t=touying.dot(u_m)

import matplotlib.pyplot as plt
plt.figure(figsize=[5,5])
plt.scatter(p[0,0],p[0,1])
plt.plot([0,m[0,0]],[0,m[0,1]])
plt.scatter(m[0,0],m[0,1])
plt.scatter(t[0,0],t[0,1])
plt.xlim(0,10)
plt.ylim(0,10)
