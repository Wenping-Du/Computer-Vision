import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.constants import degree


fig = plt.figure(figsize=(5, 5))

v = np.array([[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1]])

v2 = np.array([[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
              [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# assume a is angle

f = 1
P = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])

T = np.array([[1, 0, 0, 5],
              [0, 1, 0, 10],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

i = np.matmul(T, v)
i = np.matmul(P, i)
i = i[0:2, :] / i[2]

print(i[0, :])
print(i[1, :])

i2 = np.matmul(P, v2)
i2 = i2[0:2, :] / i2[2]

print(i2[0, :])
print(i2[1, :])

plt.scatter(i[0, :], i[1, :], c='g', marker='o')
plt.scatter(i2[0, :], i2[1, :], c='b', marker='s')

plt.xlabel('X')
plt.ylabel('Y')
plt.title("parallel lines, 5 unit right, 2 unit up translation")
# plt.title("parallel lines, translate 10 unit to left")
plt.show()