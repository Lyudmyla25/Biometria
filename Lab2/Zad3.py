import numpy as np
import cv2

img_before = cv2.imread('super_big_plums.jpg', 0)


def calc(img):
    _, thresh = cv2.threshold(img, 31, 255, cv2.THRESH_BINARY)
    rev_thresh = cv2.bitwise_not(thresh)
    ret, markers = cv2.connectedComponents(rev_thresh, connectivity=4)
    surface = np.count_nonzero(markers)
    average_size = surface / ret
    return average_size


print("Plum average size: {}".format(calc(img_before)))

kernel = np.ones((5, 5), np.float32) / 25
img_after = cv2.filter2D(img_before, -1, kernel)

print("Plum average size: {}".format(calc(img_after)))
