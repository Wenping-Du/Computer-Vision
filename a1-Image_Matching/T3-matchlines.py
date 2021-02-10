import cv2

def get_matchlines(img1, img2):
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    im3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:40], im2, flags=2)
    cv2.imwrite("/Users/deb/Documents/images/output/matchlines.png", im3)
    print("get the match lines between two images")
    # plt.imshow(img3), plt.show()
