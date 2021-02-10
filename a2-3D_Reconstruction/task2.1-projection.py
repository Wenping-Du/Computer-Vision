import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# ax3 = fig.add_subplot(223, projection='3d')
# ax4 = fig.add_subplot(224)
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
# 假设基线距离为100
P2 = np.array([[f, 0, 0, -100],
              [0, f, 0, 0],
              [0, 0, 1, 0]])

# print(i)


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
