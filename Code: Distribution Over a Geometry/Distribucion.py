import numpy as np
import matplotlib.pyplot as plt
step=0.0001
r = 2
p1 = np.array([-2, 0])  ## Here you choose the initial point of vector distribution
x = []
t = []
for xi in np.arange(-2, 2, step):
    p2 = np.abs((xi)**2 - (r**2))**(1/2)
    l = np.array([xi, p2])  
    v = np.linalg.norm(l - p1)  
    t.append(v)
for xi in np.arange(2, -2, -step):
    p2 = np.abs((xi)**2 - (r**2))**(1/2)
    l = np.array([xi, -p2]) 
    v = np.linalg.norm(l - p1)  
    t.append(v)
q=np.arange(0,8,step)
x.append(q)

plt.figure(figsize=(10, 10))
plt.plot(q,t)
plt.xlim(-1, 9)  
plt.ylim(0, 10)
plt.show()
