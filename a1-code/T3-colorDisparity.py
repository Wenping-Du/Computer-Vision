import numpy as np
from PIL import Image
import random
import cv2

def get_disparity(a, b, array1, array2, pSize):
    p = pSize // 2
    sum_a = sum_b = 0
    for m in range(2 * p + 1):
        for n in range(2 * p + 1):
            sum_a += array1[a[0] - p + m, a[1] - p + n]
            sum_b += array2[b[0] - 2 * p + m - 1, b[1] - 2 * p + n - 1]
    # print(sum_a, sum_b)
    dis = sum_b - sum_a
    return dis


def random_search(offset, a, disparity, A, B, psize):
    a_x = a[0]
    a_y = a[1]
    h = A.shape[0]
    w = B.shape[1]
    p = p_size // 2
    r = 0.5
    i = 4
    search_h = h * r ** i
    search_w = w * r ** i
    b_x = offset[a_x, a_y][0]
    b_y = offset[a_x, a_y][1]
    while search_h > 1 and search_w > 1:
        random_b_x = np.random.randint(max(b_x - search_h, p), min(b_x + search_h, h - p))
        random_b_y = np.random.randint(max(b_y - search_w, p), min(b_y + search_w, w - p))
        search_h = h * r ** i
        search_w = w * r ** i
        b = np.array([random_b_x, random_b_y])
        d = get_disparity(a, b, A, B, psize)

        if sum(d) < sum(disparity[a_x, a_y]) and b[0] < h and b[1] < w:
            disparity[a_x, a_y] = d
            offset[a_x, a_y] = b[0], b[1], np.nan
        i += 1
    return offset, disparity

def patchmatch(img1, img2, psize, times):
    h = img1.shape[0]
    w = img1.shape[1]
    # p用于防止坐标越界
    p = psize // 2

    # -----------initial offset and disparity ------------
    offset = np.ones_like(img1, dtype=object) * np.nan
    disparity = np.ones_like(img1) * np.nan
    for i in range(h):
        for j in range(w):
            a = np.array([i, j])
            b = np.array([random.randint(p, h - p), random.randint(p, w - p)])
            offset[i, j, :] = b[0], b[1], np.nan
            disparity[i, j, :] = get_disparity(a, b, img1, img2, psize)
    # print(offset)
    # print(disparity)
    # Image.fromarray(disparity.astype(np.uint8)).show()

    for it in range(times):
        print(it + 1, "start to processing...")
        for i in range(h):
            for j in range(w):
                # -----------propagation--------------------
                a = np.array([i, j])
                current = np.sum(disparity[i, j])
                if it % 2 == 0:
                    down = np.sum(disparity[i, min(j + 1, w - 1)])
                    right = np.sum(disparity[min(i + 1, h - 1), j])
                    if right < current:
                        offset[i, j, :] = min(i + 1, h - 1), j, np.nan
                        current = right
                    if down < current:
                        offset[i, j, :] = i, min(j + 1, w - 1), np.nan
                        current = down
                    if right < current:
                        offset[i, j, :] = min(i + 1, h - 1), j, np.nan
                else:
                    up = np.sum(disparity[i, max(j - 1, 0)])
                    left = np.sum(disparity[max(i - 1, 0), j])
                    if up < current:
                        offset[i, j, :] = i, max(j - 1, 0), np.nan
                        current = up
                    if left < current:
                        offset[i, j, :] = max(i - 1, 0), j, np.nan
                        current = left
                    if up < current:
                        offset[i, j, :] = i, max(j - 1, 0), np.nan
                b = np.array([offset[i, j][0], offset[i, j][1]])
                disparity[i, j, :] = get_disparity(a, b, img1, img2, psize)
                # -------------------random search-----------------------
                offset, disparity = random_search(offset, a, disparity, img1, img2, psize)
    # print(disparity)
    disImage = Image.fromarray(disparity.astype(np.uint8))
    disImage.show()

    # ----------------------------reconstruction-----------------------------------
    # img1, offset, disparity
    # print(offset)
    newImage = np.zeros_like(img2)
    for i in range(h):
        for j in range(w):
            x = offset[i, j][0]
            y = offset[i, j][1]
            if x < h and y < w:
                newImage[i, j, :] = img2[x, y, :] - disparity[i, j]
    return newImage


if __name__ == "__main__":
    array1 = np.array(Image.open('/Users/deb/Documents/images/output/im0.png'))
    array2 = np.array(Image.open('/Users/deb/Documents/images/output/im1.png'))
    p_size = 1
    itr = 1
    newimg = patchmatch(array1, array2, p_size, itr)
    Image.fromarray(newimg.astype(np.uint8)).show()