import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.constants import degree


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


fig = plt.figure(figsize=(5, 5))
v = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
              [8, 8, 8, 9, 9, 9, 10, 10, 10, 8, 8, 8, 9, 9, 9, 10, 10, 10, 8, 8, 8, 9, 9, 9, 10, 10, 10],
              [14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16,  14, 15, 16, 14, 15, 16, 14, 15, 16],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1,1,  1,  1, 1,  1,  1,  1, 1, 1]])

v2 = np.array([[22], [30], [30], [1]])
# assume a is angle

f = 1
P = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])
a = 30

# M = np.array([[math.cos(math.radians(a)), -math.sin(math.radians(a)), 0, 0],
#               [math.sin(math.radians(a)), math.cos(math.radians(a)), 0, 0],
#               [0, 0, 1, 0]])
R = np.array([[math.cos(math.radians(a)), 0, math.sin(math.radians(a)), 5],
              [0, 1, 0, 0],
              [-math.sin(math.radians(a)), 0, math.cos(math.radians(a)), 0],
              [0, 0, 0, 1]])

i = np.matmul(R, v)
i = np.matmul(P, i)
i = i[0:2, :] / i[2]

i2 = np.matmul(R, v2)
i2 = np.matmul(P, i2)
i2 = i2[0:2, :] / i2[2]

plt.scatter(i[0, :], i[1, :], c='g', marker='o')
plt.scatter(i2[0, :], i2[1, :], c='b', marker='o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("cube, translate 5 to right after rotating")
plt.show()