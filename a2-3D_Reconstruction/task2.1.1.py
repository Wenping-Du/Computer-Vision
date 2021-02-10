# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
#
# file_data_path = "/Users/deb/Downloads/sample.xyz"
# point_cloud = np.loadtxt(file_data_path,skiprows=1)
# mean_Z = np.mean(point_cloud,axis=0)[2]
# spatial_query = point_cloud[abs(point_cloud[:,2]-mean_Z)<1]
# xyz = spatial_query[:, :3]
# rgb = spatial_query[:, 3:]
# ax = plt.axes(projection='3d')
# ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb/255, s=0.01)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# !/usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__':
    # Read source image.
    im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

    # Read destination image.
    im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)
