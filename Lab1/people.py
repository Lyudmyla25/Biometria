import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def log_and_range(img, add=10):
    img_f = img.astype(float)
    img_f += add
    img_f /= 1
    img_log = np.log(img_f)

    np.min(img_log)
    np.max(img_log)
    img_full_r = (img_log - np.min(img_log)) * 255 / (np.max(img_log) - np.min(img_log))
    return img_full_r.astype(int)


img = cv.imread('kobieta.jpg')

img_increase_col = img.max(axis=2)

img_full_r = log_and_range(img)
plt.imshow(log_and_range(img_increase_col, 1))
plt.show()

plt.hist(img.ravel(), 256, [0, 256])
plt.show()
