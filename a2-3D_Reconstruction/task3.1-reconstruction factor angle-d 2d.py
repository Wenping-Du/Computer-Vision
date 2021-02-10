import math
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(111)

# cloud points
def create_cameraB(f, angle, d):
    theta = math.radians(angle / 2)
    t = math.tan(theta)
    tx = 2 * d * t / math.sqrt(1 + math.pow(t, 2))
    tz = 2 * d * math.pow(t, 2) / math.sqrt(1 + math.pow(t, 2))
    # print(tx)
    # print(tz)
    P2 = np.array([[f, 0, 0, 0],
                   [0, f, 0, 0],
                   [0, 0, 1, 0]])

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
def project_image(f, points, angle, distance):
    # --------------------------camera A  intrinsic -------------------------------
    P1 = np.array([[f, 0, 0, 0],
                  [0, f, 0, 0],
                  [0, 0, 1, 0]])
    P2 = create_cameraB(f, angle, distance)
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
# for d in range(50):
#     distance = 1 + d
for a in range(180):
    distance = 12
    angle = 1 + a
    fl = 10.0
    P1, P2, image1, image2 = project_image(fl, v, angle, distance)
    len = np.size(image1[0, :])
    original = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, image1, image2, original)
    # 未添加高斯噪声 原始点
    original = original[0:3, :] / original[3]
    # for a in range(40):
    # 图像经过高斯噪声
    img1_noise = add_gaussian_noise(image1, 5)
    img2_noise = add_gaussian_noise(image2, 5)

    reconstruct = np.zeros((4, len))
    cv2.triangulatePoints(P1, P2, img1_noise, img2_noise, reconstruct)
    reconstruct = reconstruct[0:3, :] / reconstruct[3]
    x.append(distance)
    y.append(angle)
    z.append(cal_residual(original, reconstruct))

for i in range(20):
    maxIndex = z.index(max(z))
    z.remove(max(z))
    # del x[maxIndex]
    del y[maxIndex]
#
# ax1.scatter(x, z, c='b', marker='o')
# ax1.set_xlabel('convergence distance')
# ax1.set_ylabel('average error')

ax1.scatter(y, z, c='b', marker='o')
ax1.set_xlabel('convergence angle')
ax1.set_ylabel('average error ')
plt.show()
