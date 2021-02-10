import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.constants import degree


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


fig = plt.figure(figsize=(5, 5))

v = np.array([[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1]])

v2 = np.array([[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


f = 1
P = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])
# assume a is angle
a = 45
R = np.array([[math.cos(math.radians(a)) * 0.5, 0, math.sin(math.radians(a)), 0],
              [0, 0.5, 0, 0],
              [-math.sin(math.radians(a)), 0, math.cos(math.radians(a)), 0],
              [0, 0, 0, 1]])

# the 1-st line
# [P]*[R|t]*points
i = np.matmul(R, v)
i = np.matmul(P, i)
i = i[0:2, :] / i[2]

# the 2-nd line
i2 = np.matmul(R, v2)
i2 = np.matmul(P, i2)
i2 = i2[0:2, :] / i2[2]


plt.scatter(i[0, :], i[1, :], c='g', marker='o')
plt.scatter(i2[0, :], i2[1, :], c='b', marker='s')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("parallel lines, translate 10 units to right")
plt.show()