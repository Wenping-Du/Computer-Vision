import numpy as np
import matplotlib.pyplot as plt
import math
import random


def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
n = 100
# 指定Z
total_xs = []
total_ys = []
total_zs = []
for i in range(n):
    x = i * random.random()
    xs = randrange(n, -x, x)
    ys = []
    for j in range(50):
        y = math.sqrt(x * x - math.pow(xs[j], 2))
        ys.append(y)
    for j in range(50):
        y = -math.sqrt(x * x - math.pow(xs[j - 50], 2))
        ys.append(y)
    total_xs.append(xs)
    total_ys.append(ys)
    total_zs.append(randrange(n, 5, 5))

# # 指定X
# x_total_xs = []
# x_total_ys = []
# x_total_zs = []
# for i in range(n):
#     y = i * random.random()
#     ys = randrange(n, -y, y)
#     zs = []
#     for j in range(50):
#         z = math.sqrt(y * y - math.pow(ys[j], 2))
#         zs.append(z)
#     for j in range(50):
#         z = -math.sqrt(y * y - math.pow(ys[j - 50], 2))
#         zs.append(z)
#     x_total_xs.append(randrange(n, 5, 5))
#     x_total_ys.append(ys)
#     x_total_zs.append(zs)
#
# # 指定Y
# y_total_xs = []
# y_total_ys = []
# y_total_zs = []
# for i in range(n):
#     z = i * random.random()
#     zs = randrange(n, -z, z)
#     xs = []
#     for j in range(50):
#         x = math.sqrt(z * z - math.pow(zs[j], 2))
#         xs.append(x)
#     for j in range(50):
#         x = -math.sqrt(z * z - math.pow(zs[j - 50], 2))
#         xs.append(x)
#     y_total_xs.append(xs)
#     y_total_ys.append(randrange(n, 5, 5))
#     y_total_zs.append(zs)

for m in range(n):
    if m % 2 == 0:
        ax.scatter(total_xs[m], total_ys[m], total_zs[m], c='g', marker='o')
    else:
        ax.scatter(total_xs[m], total_ys[m], total_zs[m], c='b', marker='o')
    # if m % 2 == 0:
    #     ax.scatter(x_total_xs[m], x_total_ys[m], x_total_zs[m], c='g', marker='o')
    # else:
    #     ax.scatter(x_total_xs[m], x_total_ys[m], x_total_zs[m], c='c', marker='o')
    # if m % 2 == 0:
    #     ax.scatter(y_total_xs[m], y_total_ys[m], y_total_zs[m], c='b', marker='o')
    # else:
    #     ax.scatter(y_total_xs[m], y_total_ys[m], y_total_zs[m], c='c', marker='o')
plt.title("circle image")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# plt.title("parallel lines, translate 10 unit to left")
plt.show()
