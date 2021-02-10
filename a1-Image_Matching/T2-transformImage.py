from PIL import Image
import numpy as np
import cv2


def change_image(image):
    m = (image - np.mean(image)) / np.std(image)
    output = Image.fromarray(m.astype(np.uint8))
    return output


image2 = cv2.imread('im3-0.png')
im2 = change_image(image2)

im2.show()






