import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.unicode_minus'] = False

def randrange(n, min, max):
    return np.random.rand(n)*(max - min) + min


fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
w = []
for i in range(100):
    w.append(1)

n = 100
xs = randrange(n, 0, 30)
ys = randrange(n, 10, 10)
zs = randrange(n, 10, 10)
# xs = randrange(n, 10, 20)
# ys = randrange(n, 15, 15)
# zs = 2 - 0.4 * xs

xs2 = randrange(n, 10, 40)
ys2 = randrange(n, 15, 15)
zs2 = randrange(n, 20, 20)
# xs2 = randrange(n, 10, 20)
# ys2 = randrange(n, 15, 15)
# zs2 = 8 - 0.4 * xs2

ax.scatter(xs, ys, zs,  c='g', marker='d')
ax.scatter(xs2, ys2, zs2,  c='b', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title("parallel lines")
plt.show()