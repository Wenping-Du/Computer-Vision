import math

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# ax3 = fig.add_subplot(223, projection='3d')
# ax4 = fig.add_subplot(224)
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
ax1.scatter(i[0, :], i[1, :], c='g', marker='o')

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_title("camera A")

# ---------------------------camera B  projection -------------------------------
# 矩阵乘积
ib = np.matmul(P2, v)
ib = ib[0:2, :] / ib[2]
ax2.scatter(ib[0, :], ib[1, :], c='g', marker='o')
# 参照点
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_title("camera B")
plt.show()
