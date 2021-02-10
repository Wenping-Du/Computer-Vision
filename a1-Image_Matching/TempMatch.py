import cv2
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv2.imread('/home/deb/Desktop/cv-img/im1_d.png', 0)
template = cv2.imread('/home/deb/Desktop/cv-img/im1-m-crop3x3-center.png',0)

w, h = template.shape[::-1]
res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9

loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (125,125,0), 1)
cv2.imwrite('/home/deb/Desktop/cv-img/im1-patch.png',img_rgb)
plt.imshow(img_rgb, 'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()