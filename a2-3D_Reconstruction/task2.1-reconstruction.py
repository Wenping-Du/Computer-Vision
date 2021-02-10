import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111, projection='3d')
# cloud points
v = np.array([[400, 400, 400, 400, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500, 500, 500, 500, 500, 600, 600, 600, 600, 600, 600, 600, 600, 600, 2000],
              [800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 2500],
              [1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 2500],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1, 1,  1,  1, 1,  1,  1,  1, 1, 1, 1]])
# focal length
f = 1.0
# --------------------------camera A  intrinsic -------------------------------
P1 = np.array([[f, 0, 0, 0],
              [0, f, 0, 0],
              [0, 0, 1, 0]])
# --------------------------camera B  intrinsic -------------------------------
# 假设基线距离为10
P2 = np.array([[f, 0, 0, -100],
              [0, f, 0, 0],
              [0, 0, 1, 0]])

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
img1_noise = add_Gaussian_noise(i, 2)
img2_noise = add_Gaussian_noise(ib, 2)
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