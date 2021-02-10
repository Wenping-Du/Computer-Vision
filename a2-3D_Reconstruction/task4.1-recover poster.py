import math
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

# fig = plt.figure(figsize=(20, 6))
# ax1 = fig.add_subplot(131)
# ax2 = fig.add_subplot(132)
# ax3 = fig.add_subplot(133)
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111, projection='3d')

def create_cameraB(K, angle, d):
    theta = math.radians(angle / 2)
    t = math.tan(theta)
    tx = 2 * d * t / math.sqrt(1 + math.pow(t, 2))
    tz = 2 * d * math.pow(t, 2) / math.sqrt(1 + math.pow(t, 2))

    theta = math.radians(angle / 2)
    RX = np.array([[1, 0, 0],
                   [0, math.cos(theta), -math.sin(theta)],
                   [0, math.sin(theta), math.cos(theta)]])
    T = np.array([[tx], [0], [tz]])
    return np.dot(K,  np.hstack((RX, T)))


v = np.array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 120, 120, 120, 120, 120, 120, 120, 120, 120, 250],
              [140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 300],
              [200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 350],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1, 1,  1,  1, 1,  1,  1,  1, 1, 1, 1]])

# focal length
f = 1.0
P1 = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])

K = np.array([[f, 0, 0],
              [0, f, 0],
              [0, 0, 1]])
P2 = create_cameraB(K, 60, 12)

i = np.matmul(P1, v)
i = i[0:2, :] / i[2]
ax1.scatter(i[0, :], i[1, :], c='g', marker='o')

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_title("camera A")

# ---------------------------camera B  projection -------------------------------
# 矩阵乘积
ib = np.matmul(P2, v)
ib = ib[0:2, :] / ib[2]
# ax2.scatter(ib[0, :], ib[1, :], c='g', marker='o')
# ax2.set_xlabel('X Label')
# ax2.set_ylabel('Y Label')
# ax2.set_title("camera B")

# /////////////////////////////recover pose/////////////////////////////////
point2D = (0., 0.)
points1 = np.transpose(i)
points2 = np.transpose(ib)
E, mask = cv2.findEssentialMat(points1, points2, f, point2D)
print(mask)
points, R, t, mask = cv2.recoverPose(E, points1, points2)

print("R =", R)
print("t =", t)
# print(mask)
P3 = np.dot(K, np.hstack((R, t)))
# print(P3)

ib3 = np.dot(P3, v)
ib3 = ib3[0:2, :] / ib3[2]
# ax3.scatter(ib3[0, :], ib3[1, :], c='g', marker='o')

len = np.size(i[0, :])
original = np.zeros((4, len))
cv2.triangulatePoints(P1, P2, i, ib, original)
original = original[0:3, :] / original[3]

reconstruct = np.zeros((4, len))
cv2.triangulatePoints(P1, P2, i, ib3, reconstruct)
reconstruct = reconstruct[0:3, :] / reconstruct[3]


# ax3.set_xlabel('X Label')
# ax3.set_ylabel('Y Label')
# ax3.set_title("camera B recover")
# plt.show()

# green is original, blue is reconstructed
ax1.scatter(original[0, :], original[1, :], original[2, :],  c='g', marker='o')
ax1.scatter(reconstruct[0, :], reconstruct[1, :], reconstruct[2, :],  c='b', marker='o')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title("Reconstruction using recoverpose")
plt.show()