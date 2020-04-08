import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def rozciaganie(hist):
    minimum = 0
    maximum = 255
    LUT = [0] * 256
    while hist[minimum] <= 0:
        minimum = minimum + 1
    while hist[maximum] <= 0:
        maximum = maximum - 1
    for i in range(0, 256):
        LUT[i] = (255 / (maximum - minimum)) * (i - minimum)
    return LUT


def wyrownanie(hist, img):
    r = img.shape[0] * img.shape[1]
    LUT = [0] * 256
    D = [0] * 256
    for i in range(0, 256):
        j = 0
        for k in range(0, i + 1):
            j = j + hist[k]
        D[i] = j / r
    n = 0
    while D[n] <= 0:
        n = n + 1
    minD = D[n]
    for i in range(0, 256):
        LUT[i] = ((D[i] - minD) / (1 - minD)) * 255
    return LUT


def zmiana_obrazka(img):
    histogram = cv.calcHist([img], [0], None, [256], [0, 256])
    LUT = rozciaganie(histogram)

    new_image_r = [[LUT[img[i][j]] for j, _ in enumerate(img[0])] for i, _ in enumerate(img)]
    new_image_r = np.asarray(new_image_r, dtype=np.uint8)

    histogram = cv.calcHist([new_image_r], [0], None, [256], [0, 256])
    LUT = wyrownanie(histogram.flatten(), new_image_r)

    new_image_w = [[LUT[new_image_r[i][j]] for j, _ in enumerate(new_image_r[0])] for i, _ in enumerate(new_image_r)]
    new_image_w = np.asarray(new_image_w)
    return new_image_w


img = cv.imread('contrast.jpg', 0)
plt.imshow(img, cmap='gray')
plt.show()

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

img_lewa = img[0:img.shape[0], 0:int(img.shape[1] / 2)]
img_prawa = img[0:img.shape[0], int(img.shape[1] / 2):img.shape[1]]

img_lewa_po = zmiana_obrazka(img_lewa)
img_prawa_po = zmiana_obrazka(img_prawa)

gotowy_obrazek = np.concatenate([img_lewa_po, img_prawa_po], axis=1)
plt.imshow(gotowy_obrazek, cmap='gray')
plt.show()

plt.hist(gotowy_obrazek.ravel(), 256, [0, 256])
plt.show()
