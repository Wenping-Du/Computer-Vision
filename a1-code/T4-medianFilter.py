import cv2
import numpy as np
from PIL import Image

def mean_filtering(image, window_size):
    h = image.shape[0]
    w = image.shape[1]
    img = image.copy()
    mid = (window_size - 1) // 2
    aa = []
    for i in range(h - window_size):
        for j in range(w - window_size):
            for m1 in range(window_size):
                for m2 in range(window_size):
                    aa.append(int(image[i + m1, j + m2]))

            aa.sort()
            img[i+mid, j+mid] = aa[(len(aa)+1) // 2]
            del aa[:]
    cv2.imwrite('/Users/deb/Documents/images/output/ppp1.png', img)


# img = cv2.imread("/Users/deb/Documents/images/output/temg_square.png")  # 读取目标图片
# mean_img = cv2.medianBlur(img, 5)
#
# cv2.imshow("dd", mean_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = Image.open('/Users/deb/Documents/images/output/tmpg_square.png')
i = np.array(img)
mean_filtering(i, 2)

