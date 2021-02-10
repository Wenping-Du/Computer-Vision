import cv2


def down_sample(input_img, output_img):
    gauss0 = cv2.imread(input_img)
    # gauss
    gauss1 = cv2.pyrDown(gauss0)
    gauss2 = cv2.pyrDown(gauss1)
    gauss3 = cv2.pyrDown(gauss2)

    cv2.imwrite(output_img, gauss3)
    print(input_img, "down sampling image...")
    return output_img


def get_overlop(img1, img2):
    img1_1 = cv2.imread(img1)
    img2_1 = cv2.imread(img2)
    overlapping = cv2.addWeighted(img1_1, 0.5, img2_1, 0.5, 0)
    cv2.imwrite('/Users/deb/Documents/images/output/im_overlop.png', overlapping)


img1 = down_sample("/Users/deb/Documents/images/im0.png",
                   "/Users/deb/Documents/images/im0_out.png")

img2 = down_sample("/Users/deb/Documents/images/im1L.png",
                   "/Users/deb/Documents/images/im1L_out.png")

get_overlop(img1, img2)