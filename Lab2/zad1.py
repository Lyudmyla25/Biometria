import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('super_big_eagle.jpg')
img2 = cv.imread('super_big_lake.jpg')


def add_images(img1, img2, w=0.5):
    return (w*img1+(1-w)*img2).astype(int)


result_img = add_images(img1, img2)
plt.imshow(result_img)
plt.show()
cv.imwrite("img_sum.jpg", result_img)
