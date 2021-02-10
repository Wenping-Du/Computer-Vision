# import pillow
from PIL import Image

import numpy as np
from PIL import Image
from PIL import ImageDraw
import os
import cv2

input_img = "/home/deb/Desktop/cv-img/im1_d.png"
# # 生成格网
#
(filepath,filename) = os.path.split(input_img)
img = Image.open(input_img)
# img_d = ImageDraw.Draw(img)
# x_len, y_len = img.size
# x_step = x_len/25
# y_step = y_len/15
# for x in range(0, x_len, int(x_step)):
#     img_d.line(((x, 0), (x, y_len)), (0, 125, 0))
# for y in range(0, y_len, int(y_step)):
#     j = y_len - y - 1
#     img_d.line(((0, j), (x_len, j)), (0, 125, 0))
# img.save(os.path.join(filepath,"grid_"+filename))




# crop 3*3 pixel
# img2 = img.crop((0, 0, 3, 3))
# img2.save("/home/deb/Desktop/cv-img/im1-m-crop3x3.png")


half_the_width = img.size[0] / 2 - 20
half_the_height = img.size[1] / 2 -10
img4 = img.crop(
    (
        half_the_width - 10,
        half_the_height - 10,
        half_the_width + 10,
        half_the_height + 10
    )
)
img4.save("/home/deb/Desktop/cv-img/im1-m-crop3x3-center.png")