from PIL import Image
import numpy as np


def cal_intensity(x, y, array1, array2, pSize):

    p = pSize // 2
    # calculate pa's sum
    sum_a = 0
    for m in range(2 * p + 1):
        for n in range(2 * p + 1):
            sum_a += array1[x - p + m, y - p + n]
    B_w = np.size(array2, 1)
    disparity = {}
    # assume i == x
    for j in range(B_w):
        sum_b = 0
        for m in range(2 * p + 1):
            for n in range(2 * p + 1):
                if j - p > 0 and j + p < B_w:
                    sum_b += array2[x - p + m, j - p + n]

        tem = np.square(sum_b - sum_a)
        disparity[j] = tem
    print(disparity)
    return disparity


im1 = Image.open('/Users/deb/Documents/images/output/im3-0.png').convert('L')
im2 = Image.open('/Users/deb/Documents/images/output/im3-1E.png').convert('L')

# intensity array
array1 = np.array(im1)
array2 = np.array(im2)

patch_size = 1
x = 120
y = 61
disparity = cal_intensity(x, y, array1, array2, patch_size)
idx = np.argmin(disparity)
print(idx)

