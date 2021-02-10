import math
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111, projection='3d')
# cloud points
def create_cameraB(f, angle):
    tx = 12
    tz = 4 * math.pow(3, 0.5)
    P2 = np.array([[f, 0, 0, 0],
                   [0, f, 0, 0],
                   [0, 0, 1, 0]])
    theta = math.radians(angle/2)
    RX = np.array([[1, 0, 0, 0],
                   [0, math.cos(theta), -math.sin(theta), 0],
                   [0, math.sin(theta), math.cos(theta), 0],
                   [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, 0],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])
    return np.matmul(P2, np.matmul(T, RX))


v = np.array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 120, 120, 120, 120, 120, 120, 120, 120, 120, 250],
              [140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 300],
              [200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 350],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1, 1,  1,  1, 1,  1,  1,  1, 1, 1, 1]])

# focal length
f = 1.0
# --------------------------camera A  intrinsic -------------------------------
P1 = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])
# --------------------------camera B  projection matrix -------------------------------
P2 = create_cameraB(f, 60)

# --------------------------camera A  projection -------------------------------
# 矩阵乘积
i = np.matmul(P1, v)
i = i[0:2, :] / i[2]

# ---------------------------camera B  projection -------------------------------
# 矩阵乘积
ib = np.matmul(P2, v)
ib = ib[0:2, :] / ib[2]

# -------------------------------Reconstruction---------------------------------
# 未添加高斯噪声 原始点
len = np.size(i[0, :])
original = np.zeros((4, len))
cv2.triangulatePoints(P1, P2, i, ib, original)
original = original[0:3, :] / original[3]

# 添加高斯随机噪声
# 随机均值mean 标准差std 数量amount
def add_Gaussian_noise(image_in, noise_sigma):
    return image_in + np.random.normal(0, 0.0001 * noise_sigma, image_in.shape)


# 图像经过高斯噪声
img1_noise = add_Gaussian_noise(i, 20)
img2_noise = add_Gaussian_noise(ib, 20)
reconstruct = np.zeros((4, len))
cv2.triangulatePoints(P1, P2, img1_noise, img2_noise, reconstruct)
reconstruct = reconstruct[0:3, :] / reconstruct[3]

# green is original, blue is reconstructed
ax1.scatter(original[0, :], original[1, :], original[2, :],  c='g', marker='o')
ax1.scatter(reconstruct[0, :], reconstruct[1, :], reconstruct[2, :],  c='b', marker='o')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title("Reconstruction with Gaussian noise")
plt.show()