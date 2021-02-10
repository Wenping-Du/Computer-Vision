import numpy as np
import matplotlib.pyplot as plt

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
# original x = ±1, y = ±1, z = 1, w = 1
v = np.array([[1,  1, -1, -1],
              [1, -1,  1, -1],
              [1,  1,  1,  1],
              [1,  1,  1,  1]])

# new for test x = ±1, y = −1, z = 2 ± 1, w = 1
# v = np.array([[1,  1, -1, -1],
#               [-1, -1, -1, -1],
#               [3,  1,  3,  1],
#               [1,  1,  1,  1]])
# print(v)
# focal length

# fig = plt.figure(figsize=(5, 5))
# # plt.scatter(i[0, :], i[1, :], c='g', marker='D')
# plt.scatter(i[0, 0], i[1, 0], c='k', marker='D')
# plt.scatter(i[0, 1], i[1, 1], c='b', marker='D')
# plt.scatter(i[0, 2], i[1, 2], c='r', marker='D')
# plt.scatter(i[0, 3], i[1, 3], c='g', marker='D')

for i in range(4):
    if i == 0:
        c = 'k'
    if i == 1:
        c = 'b'
    if i == 2:
        c = 'r'
    if i == 3:
        c = 'g'
    ax.scatter(v[0, i], v[1, i], v[2, i], c=c, marker='d')
plt.title("1st set")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
