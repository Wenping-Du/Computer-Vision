import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.unicode_minus'] = False

v = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
              [8, 8, 8, 9, 9, 9, 10, 10, 10, 8, 8, 8, 9, 9, 9, 10, 10, 10, 8, 8, 8, 9, 9, 9, 10, 10, 10],
              [14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16, 14, 15, 16,  14, 15, 16, 14, 15, 16, 14, 15, 16],
              [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1, 1, 1,1,  1,  1, 1,  1,  1,  1, 1, 1]])

# v = np.array([[-3.102, -3.102, -1.27200007, -1.27200007, -3.102, -3.102 , -1.27200007, -1.27200007, ],
#               [-1.58400011, -0.08400005, -0.08400005 , -1.58400011, -1.58400011, -0.08400005, -0.08400005, -1.58400011],
#               [9.29399872, 9.2939987, 9.29399872, 9.29399872 ,13.8939991, 13.8939991, 13.8939991, 13.8939991],
#               [1,  1,  1,  1, 1,  1,  1,  1, ]])
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

xs = v[0]
ys = v[1]
zs = v[2]

ax.scatter(xs, ys, zs,  c='g', marker='o')

v2 = np.array([[22], [30], [30], [1]])
xs2 = v2[0]
ys2 = v2[1]
zs2 = v2[2]
ax.scatter(xs2, ys2, zs2,  c='b', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title("cube")
plt.show()