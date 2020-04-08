import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab4/retina.jpg', -1)

grey_average = np.mean(img, 2)
grey_red_level = img[:, :, 2]
grey_human = 0.1140 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.2989 * img[:, :, 2]
cv2.imwrite("Lab4/grey_average.png", grey_average)
cv2.imwrite("Lab4/grey_red_level.png", grey_red_level)
cv2.imwrite("Lab4/grey_human.png", grey_human)

plt.hist(grey_average.ravel(), 256, [3, 256])
plt.title('Grey Average')
plt.show()

plt.hist(grey_human.ravel(), 256, [3, 256])
plt.title('Grey Human')
plt.show()

plt.hist(grey_red_level.ravel(), 256, [3, 256])
plt.title('Grey Red Level')
plt.show()
