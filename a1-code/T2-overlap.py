# coding=utf-8
import cv2

# 底板图案
img1 = '/Users/deb/Documents/images/output/im3-0.png'
# 上层图案
img2 = '/Users/deb/Documents/images/output/im3-1.png'


img1_1 = cv2.imread(img1)
img2_1 = cv2.imread(img2)
# 权重越大，透明度越低
overlapping = cv2.addWeighted(img1_1, 0.5, img2_1, 0.5, 0)
# 保存叠加后的图片
cv2.imwrite('/Users/deb/Documents/images/output/im3_overlop.png', overlapping)