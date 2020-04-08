import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Lab2/super_big_Lenna.png')


def salt_noise(img, p=0.01):
    noise_M = np.random.choice((-255, 255, 0), img.shape[:2], p=(p / 2, p / 2, 1 - p))
    result = img.copy().astype('int64')
    if img.ndim == 2:
        result += noise_M
    if img.ndim == 3:
        for dim in range(3):
            result[:, :, dim] += noise_M
    return result.clip(0, 255).astype('uint8')


img_noise = salt_noise(img, 0.01)

plt.imshow(img_noise)
plt.show()
cv.imwrite("Lab2/Figure_1.png", img_noise)