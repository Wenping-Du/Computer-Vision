import math
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(122)
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



# focal length
def project_image(f, points):
    # --------------------------camera A  intrinsic -------------------------------
    P1 = np.array([[f, 0, 0, 0],
                  [0, f, 0, 0],
                  [0, 0, 1, 0]])
    # --------------------------camera B  intrinsic -------------------------------
    # 假设基线距离为10
    P2 = create_cameraB(f, 60)
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
v = np.array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 120, 120, 120, 120, 120, 120, 120, 120, 120, 250],
              [140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 140, 140, 140, 150, 150, 150, 160, 160, 160, 300],
              [200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 200, 210, 220, 350],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1, 1,  1,  1, 1,  1,  1,  1, 1, 1, 1]])

x = []
y = []
z = []
for t in range(200):
    # focal length
    # fl = 1.0 + t * 0.05
    fl = 1.0
    P1, P2, image1, image2 = project_image(fl, v)
    len = np.size(image1[0, :])
    original = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, image1, image2, original)
    # 未添加高斯噪声 原始点
    original = original[0:3, :] / original[3]
    # 图像经过高斯噪声
    img1_noise = add_gaussian_noise(image1, t)
    img2_noise = add_gaussian_noise(image2, t)

    reconstruct = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, img1_noise, img2_noise, reconstruct)
    reconstruct = reconstruct[0:3, :] / reconstruct[3]

    x.append(t)
    z.append(cal_residual(original, reconstruct))
# plt.plot(x, z, "b--", linewidth=1)
ax1.scatter(x, z, c='b', marker='o')
ax1.set_xlabel('noise amount')
ax1.set_ylabel('reconstruction average error')

plt.show()
