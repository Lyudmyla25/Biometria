import numpy as np
import cv2
from skimage.color import rgb2gray
from sklearn.cluster import KMeans

img = cv2.imread('Lab7/GroupPhoto-e1577801886654-630x557.jpg')
img_grey = cv2.imread('Lab7/GroupPhoto-e1577801886654-630x557.jpg', 0)

img_grey_r = img_grey.reshape(img_grey.shape[0] * img_grey.shape[1])
for i in range(img_grey_r.shape[0]):
    if img_grey_r[i] > img_grey_r.mean():
        img_grey_r[i] = 1
    else:
        img_grey_r[i] = 0
img_grey = img_grey_r.reshape(img_grey.shape[0], img_grey.shape[1])

region_based_segmentation = rgb2gray(img)
gray_r = region_based_segmentation.reshape(region_based_segmentation.shape[0] * region_based_segmentation.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
region_based_segmentation = gray_r.reshape(region_based_segmentation.shape[0], region_based_segmentation.shape[1])

pic_n = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
kmeans = KMeans(n_clusters=3, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])

edges = cv2.Canny(img, 100, 200)
_, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh_local = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# cv2.imshow('after', cluster_pic.astype("uint8"))
cv2.imshow('edges', edges)
# cv2.imshow('thresh_global', thresh_global)
cv2.imshow('thresh_local', thresh_local)
# cv2.imshow('region_based_segmentation', region_based_segmentation)

cv2.waitKey(0)
cv2.destroyAllWindows()
