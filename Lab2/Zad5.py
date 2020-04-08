import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Lab2/super_big_Lenna.png')


def unif_dist(img, p=0.01, level = 15):
    noise_M = np.random.uniform(-level, level, img.shape[:2]).astype(int)
    noise_M *= np.random.choice((0,1), noise_M.shape, p = (1-p,p))
    result = img.copy().astype('int64')
    if img.ndim == 2:
        result += noise_M
    if img.ndim == 3:
        for dim in range(3):
            result[:, :, dim] += noise_M
    return result.clip(0, 255).astype('uint8')


img_noise = unif_dist(img, 0.30, 100)

plt.imshow(img_noise)
plt.show()
cv.imwrite("Lab2/Figure_2.png", img_noise)