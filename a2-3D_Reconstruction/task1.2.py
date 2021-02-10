import numpy as np
import matplotlib.pyplot as plt
import math


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


fig = plt.figure(figsize=(5, 5))

# v = np.array([[1,  1, -1, -1],
#               [1, -1,  1, -1],
#               [1,  1,  1,  1],
#               [1,  1,  1,  1]])
# assume a is angle
a = 30
M = np.array([[math.cos(math.radians(a)), 0, math.sin(math.radians(a)), 0],
              [0, 1, 0, 0],
              [-math.sin(math.radians(a)), 0, math.cos(math.radians(a)), 0]])
w = []
for i in range(100):
    w.append(1)

n = 100
xs = randrange(n, 0, 30)
ys = randrange(n, 10, 10)
zs = randrange(n, 10, 10)


xs2 = randrange(n, 10, 40)
ys2 = randrange(n, 15, 15)
zs2 = randrange(n, 20, 20)
# xs = randrange(n, 0, 10)
# ys = randrange(n, 10, 10)
# zs = 2 - 0.4 * xs
#
# xs2 = randrange(n, 10, 20)
# ys2 = randrange(n, 15, 15)
# zs2 = 8 - 0.4 * xs2

v = np.array([xs, ys, zs, w])
i = np.matmul(M, v)
i = i[0:2, :] / i[2]


v2 = np.array([xs2, ys2, zs2, w])
i2 = np.matmul(M, v2)
i2 = i2[0:2, :] / i2[2]

plt.scatter(i[0, :], i[1, :], c='g', marker='o')
plt.scatter(i2[0, :], i2[1, :], c='b', marker='s')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("parallel lines, rotate 45")
plt.show()