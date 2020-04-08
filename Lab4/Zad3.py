import cv2
import numpy as np

img = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab4/retina.jpg', 0)

ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(thr, kernel, iterations=1)

cv2.imwrite('Lab4/mask.png', mask)

