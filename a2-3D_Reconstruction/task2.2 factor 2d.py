import math
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(122)
# cloud points
v = np.array([[400, 400, 400, 400, 400, 400, 400, 400, 400, 500, 500, 500, 500, 500, 500, 500, 500, 500, 600, 600, 600, 600, 600, 600, 600, 600, 600, 2000],
              [800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 800, 800, 800, 900, 900, 900, 1000, 1000, 1000, 2500],
              [1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 1400, 1500, 1600, 2500],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1, 1,  1,  1, 1,  1,  1,  1, 1, 1, 1]])


# focal length
def project_image(f, points, baseline):
    # --------------------------camera A  intrinsic -------------------------------
    P1 = np.array([[f, 0, 0, 0],
                  [0, f, 0, 0],
                  [0, 0, 1, 0]])
    # --------------------------camera B  intrinsic -------------------------------
    # 假设基线距离为10
    P2 = np.array([[f, 0, 0, baseline],
                  [0, f, 0, 0],
                  [0, 0, 1, 0]])
    # --------------------------camera A  projection -------------------------------
    # 矩阵乘积 图像1
    i = np.matmul(P1, points)
    i = i[0:2, :] / i[2]
    # ---------------------------camera B  projection -------------------------------
    # 矩阵乘积 图像2
    # ib = np.matmul(t, points)
    ib = np.matmul(P2, points)
    ib = ib[0:2, :] / ib[2]
    return P1, P2, i, ib

# 添加高斯随机噪声
# 0 标准差0.01 image.shape
def add_gaussian_noise(image_in, amount):
    return image_in + np.random.normal(0, 0.0001 * amount, image_in.shape)


def cal_residual(original, reconstruct):
    size = np.size(original[0, :])
    residual = []
    for m in range(size):
        dx = math.pow(original[0, m] - reconstruct[0, m], 2)
        dy = math.pow(original[1, m] - reconstruct[1, m], 2)
        dz = math.pow(original[2, m] - reconstruct[2, m], 2)
        residual.append(math.sqrt(dx + dy + dz))
    return np.mean(residual)


# -------------------------------Relationship---------------------------------

x = []
y = []
z = []
for t in range(200):
    # focal length
    fl = 1.0 + t
    baseline = - t
    P1, P2, image1, image2 = project_image(fl, v, baseline)
    len = np.size(image1[0, :])
    original = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, image1, image2, original)
    # 未添加高斯噪声 原始点
    original = original[0:3, :] / original[3]
    # 图像经过高斯噪声
    img1_noise = add_gaussian_noise(image1, 1)
    img2_noise = add_gaussian_noise(image2, 1)

    reconstruct = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, img1_noise, img2_noise, reconstruct)
    reconstruct = reconstruct[0:3, :] / reconstruct[3]

    x.append(fl)
    z.append(cal_residual(original, reconstruct))
plt.plot(x, z, "b--", linewidth=1)
ax1.set_xlabel('focal length ')
ax1.set_ylabel('reconstruction average error')

plt.show()
