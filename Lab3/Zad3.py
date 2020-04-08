import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

img = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab3/circles.png', 0)

plt.imshow(img)
plt.show()

kernel1 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((9, 9), np.uint8)
kernelDiff = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')

img1 = cv2.erode(img, kernel1)
img2 = cv2.erode(img, kernel4)
img3 = cv2.erode(img, kernelDiff)

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

plt.imshow(img3)
plt.show()

skeleton = skeletonize(img.clip(0, 1))

plt.imshow(skeleton)
plt.show()
