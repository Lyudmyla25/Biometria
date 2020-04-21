import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg

img = cv2.imread('oko01.png', 0)


def filtration(img, matrix):
    if img.ndim == 2:
        return sg.convolve2d(img, matrix[:, ::-1], "valid").astype("uint8")
    if img.ndim == 3:
        res_lst = [sg.convolve2d(img[:, :, x], matrix[:, ::-1], "valid") for x in range(img.shape[2])]
        return np.rollaxis(np.array(res_lst), 0, 3).astype("uint8")


gauss_matrix = np.array([[0.037, 0.039, 0.04, 0.039, 0.037],
                         [0.039, 0.042, 0.042, 0.042, 0.039],
                         [0.04, 0.042, 0.043, 0.042, 0.04],
                         [0.039, 0.042, 0.042, 0.042, 0.039],
                         [0.037, 0.039, 0.04, 0.039, 0.037]])

# img = filtration(img, gauss_matrix)

kernel = np.ones((3, 3), np.uint8)
_, bin_img_teczowka = cv2.threshold(img, img.mean() / 1.5, 255, cv2.THRESH_BINARY)
_, bin_img_zrenica = cv2.threshold(img, img.mean() / 4.5, 255, cv2.THRESH_BINARY)

bin_img_teczowka = cv2.morphologyEx(bin_img_teczowka, cv2.MORPH_CLOSE, kernel, iterations=8)
bin_img_zrenica = cv2.morphologyEx(bin_img_zrenica, cv2.MORPH_CLOSE, kernel, iterations=2)

M = cv2.moments(255 - bin_img_zrenica)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
contours, hierarchy = cv2.findContours(bin_img_zrenica, 1, 2)
cnt = contours[2]
# cnt = contours[1]
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img = cv2.imread('oko01.png')
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
print(radius)
contours, _ = cv2.findContours(bin_img_teczowka, 1, 2)
cnt = contours[1]
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
print(radius)

img = cv2.circle(img, center, radius, (0, 0, 255), 2)
#first option
img2 = cv2.linearPolar(img, center=(x, y), maxRadius=52, flags=cv2.WARP_FILL_OUTLIERS)
#second option
img3 = cv2.logPolar(img, center=(x, y), M=52, flags=cv2.WARP_FILL_OUTLIERS)

plt.imshow(img)
plt.show()

plt.imshow(img2)
plt.show()

plt.imshow(img3)
plt.show()
