import numpy as np
import matplotlib.pyplot as plt
import cv2

# original x = ±1, y = ±1, z = 1, w = 1
# v = np.array([[1,  1, -1, -1],
#               [1, -1,  1, -1],
#               [1,  1,  1,  1],
#               [1,  1,  1,  1]])

# new for test x = ±1, y = −1, z = 2 ± 1, w = 1
v = np.array([[1,  1, -1, -1],
              [-1, -1, -1, -1],
              [3,  1,  3,  1],
              [1,  1,  1,  1]])
# print(v)
# focal length
f = 1
P = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])



i = np.matmul(P, v)
i = i[0:2, :] / i[2]
print(i[0, :])
print(i[1, :])
fig = plt.figure(figsize=(5, 5))
# # plt.scatter(i[0, :], i[1, :], c='g', marker='D')
plt.scatter(i[0, 0], i[1, 0], c='k', marker='D')
plt.scatter(i[0, 1], i[1, 1], c='b', marker='D')
plt.scatter(i[0, 2], i[1, 2], c='r', marker='D')
plt.scatter(i[0, 3], i[1, 3], c='g', marker='D')

plt.title("2nd set, focal length = 1")

plt.show()
