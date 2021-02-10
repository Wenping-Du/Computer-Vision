import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
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


# 矩阵乘积
i = np.matmul(P1, v)
i = i[0:2, :] / i[2]
ax1.scatter(i[0, :], i[1, :], c='g', marker='o')
# ---------------------------camera B  projection -------------------------------
# 矩阵乘积
ib = np.matmul(P2, v)
ib = ib[0:2, :] / ib[2]
ax2.scatter(ib[0, :], ib[1, :], c='g', marker='o')
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
img1_noise = add_Gaussian_noise(i, 5)
img2_noise = add_Gaussian_noise(ib, 5)

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