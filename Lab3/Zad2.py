import cv2
import numpy as np

img1 = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab3/rice-bw-2.png', 0)
img2 = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab3/count-the-numbers-for-each-object.jpg', 0)

img2 = img2[5:445, 5:480]


kernel = np.ones((5, 5), np.uint8)

img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

img2 = 255 - (img2.clip(254, 255)-254)*255

ret1, _ = cv2.connectedComponents(img1)
ret2, markets = cv2.connectedComponents(img2)

print('Number of objects in img1', ret1-1)
print('Number of objects in img2', ret2-1)
