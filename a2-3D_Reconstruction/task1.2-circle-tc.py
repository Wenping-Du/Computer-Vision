import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.constants import degree
import random

def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin

n = 100
# 指定Z
total_xs = []
total_ys = []
total_zs = []
for i in range(n):
    x = i * random.random()
    xs = randrange(n, -x, x)
    ys = []
    for j in range(50):
        y = math.sqrt(x * x - math.pow(xs[j], 2))
        ys.append(y)
    for j in range(50):
        y = -math.sqrt(x * x - math.pow(xs[j - 50], 2))
        ys.append(y)
    total_xs.append(xs)
    total_ys.append(ys)
    total_zs.append(randrange(n, 5, 5))

# v = np.array([x, y, z, w])
a = 45
M = np.array([[math.cos(math.radians(a)) * 0.5, -math.sin(math.radians(a)), 0, 20],
              [math.sin(math.radians(a)), math.cos(math.radians(a)) * 0.2, 0, -10],
              [0, 0, 1, 0]])

# M = np.array([[math.cos(math.radians(a)), 0, math.sin(math.radians(a)), 10],
#               [0, 1, 0, 0],
#               [-math.sin(math.radians(a)), 0, math.cos(math.radians(a)), 0]])
w = []
for i in range(100):
    w.append(1)

for t in range(100):

    v = np.array([total_xs[t], total_ys[t], total_zs[t], w])
    i = np.matmul(M, v)
    i = i[0:2, :] / i[2]
    if t % 2 == 0:
        plt.scatter(i[0, :], i[1, :], c='g', marker='o')
    else:
        plt.scatter(i[0, :], i[1, :], c='b', marker='o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("circle, translate after rotating")

# plt.title("parallel lines, translate 10 unit to left")
plt.show()