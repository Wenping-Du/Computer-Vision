import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
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

# ---------------------Two cameras Homography Matrix----------------------------
# image1 = []
# image2 = []

# for t in range(len):
#     image1.append([i[0, t], i[1, t]])
#     image2.append([ib[0, t], ib[1, t]])
#
# H1 = cv2.findHomography(np.array(image2), np.array(image1))
# -------------------------------Reconstruction---------------------------------
# 未添加高斯噪声 原始点
len = np.size(i[0, :])
original = np.zeros((4, len))
cv2.triangulatePoints(P1, P2, i, ib, original)
original = original[0:3, :] / original[3]

# 添加高斯随机噪声
# 随机均值0 标准差0.001 noise_sigma
def add_Gaussian_noise(image_in, noise_sigma):
    return image_in + np.random.normal(0, 0.0001 * noise_sigma, image_in.shape)


# 图像经过高斯噪声
img1_noise = add_Gaussian_noise(i, 2)
img2_noise = add_Gaussian_noise(ib, 2)

# 2D 噪声图像
ax1.scatter(img1_noise[0, :], img1_noise[1, :], c='b', marker='o')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title("Image1 with gaussian noise")
ax2.scatter(img2_noise[0, :], img2_noise[1, :], c='b', marker='o')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title("Image2 with gaussian noise")

plt.show()