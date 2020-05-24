import copy
import numpy as np
import cv2 as cv
from scipy import spatial
from math import isclose
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# Constants
KERNEL = np.ones([4, 4], np.uint8)

# Functions
def get_middle(point_a, point_b):
    return (int((point_a[0] + point_b[0]) / 2), int((point_a[1] + point_b[1]) / 2))


def get_orthogonal(point_a, point_b):
    xdif = point_a[0] - point_b[0]
    ydif = point_a[1] - point_b[1]
    point_c = (int(point_b[0] - ydif), int(point_b[1] + xdif))
    point_d = (int(point_b[0] + ydif), int(point_b[1] - xdif))
    return point_c, point_d


def get_coef(a, b):
    k = (a[1] - b[1]) / (a[0] - b[0])
    b = a[1] - int(k * a[0])
    return k, b


def find_inters_points(points, k, b):
    found = []
    for point in points:
        if isclose(point[1], k * point[0] + b, abs_tol=1):
            found.append(point)
    return found


def find_inters_points2(img, x, k, b, prev, step=1):
    found = []
    x = x + step
    y = int(x * k + b)
    while 0 < x < img.shape[1] and 0 < y < img.shape[0]:
        if img[y, x] == prev:
            prev = img[y, x]
            x = x + step
            y = int(x * k + b)
        else:
            break
    found.append((x, y))
    return found


img = cv.imread('s1.png')
aug = iaa.MeanShiftBlur()
img = aug.augment_image(img)
aug = iaa.DirectedEdgeDetect(alpha=0.3, direction=0)
img = aug.augment_image(img)
aug = iaa.MeanShiftBlur()
img = aug.augment_image(img)
aug = iaa.Emboss(alpha=0.6, strength=2.2)
img = aug.augment_image(img)
aug = iaa.MeanShiftBlur()
img = aug.augment_image(img)
aug = iaa.pillike.FilterEdgeEnhanceMore()
img2 = aug.augment_image(img)
cv.imwrite('result3.png', img2)

gray = img[:, :, 2] - img[:, :, 1]
gray[gray < 0] = 0

_, tresh = cv.threshold(gray, 55, 255, cv.THRESH_BINARY)

mask = cv.dilate(tresh, KERNEL, iterations=6)
cv.imwrite('result1.png', mask)
contours, hierarchy = cv.findContours(copy.deepcopy(mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
areas = [cv.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]
points = cnt[:, 0, :]
candidates = points[spatial.ConvexHull(points).vertices]
dist_mat = spatial.distance_matrix(candidates, candidates)
i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

point_a = (candidates[i][0], candidates[i][1])
point_b = (candidates[j][0], candidates[j][1])

point_middle = get_middle(point_a, point_b)
middle_ort1, middle_ort2 = get_orthogonal(point_a, point_middle)
middle_k, middle_b = get_coef(middle_ort1, middle_ort2)
middle_inters_points_r, = find_inters_points2(mask, point_middle[0], middle_k, middle_b,
                                              mask[point_middle[1], point_middle[0]], 1)
middle_inters_points_l, = find_inters_points2(mask, point_middle[0], middle_k, middle_b,
                                              mask[point_middle[1], point_middle[0]], -1)
middle_inters_points = [middle_inters_points_l] + [middle_inters_points_r]
for point in middle_inters_points:
    cv.circle(img, (point[0], point[1]), 4, (255, 0, 0), -1)

first_q = get_middle(point_a, point_middle)
first_q1, first_q2 = get_orthogonal(point_middle, first_q)
first_qk, first_qb = get_coef(first_q1, first_q2)
first_inters_points_r, = find_inters_points2(mask, first_q[0], first_qk, first_qb, mask[first_q[1], first_q[0]], 1)
first_inters_points_l, = find_inters_points2(mask, first_q[0], first_qk, first_qb, mask[first_q[1], first_q[0]], -1)
first_inters_points = [first_inters_points_l] + [first_inters_points_r]
for point in first_inters_points:
    cv.circle(img, (point[0], point[1]), 4, (255, 0, 0), -1)

third_q = get_middle(point_b, point_middle)
third_q1, third_q2 = get_orthogonal(point_middle, third_q)
third_qk, third_qb = get_coef(third_q1, third_q2)
third_inters_points_r, = find_inters_points2(mask, third_q[0], third_qk, third_qb, mask[third_q[1], third_q[0]], 1)
third_inters_points_l, = find_inters_points2(mask, third_q[0], third_qk, third_qb, mask[third_q[1], third_q[0]], -1)
third_inters_points = [third_inters_points_l] + [third_inters_points_r]
for point in third_inters_points:
    cv.circle(img, (point[0], point[1]), 4, (255, 0, 0), -1)

cv.line(img, point_a, point_b, (0, 0, 255), 2)

cv.line(img, point_middle, third_inters_points_l, (0, 0, 255), 2)
cv.line(img, point_middle, third_inters_points_r, (0, 0, 255), 2)

cv.line(img, point_middle, middle_inters_points_l, (0, 0, 255), 2)
cv.line(img, point_middle, middle_inters_points_r, (0, 0, 255), 2)

cv.line(img, point_middle, first_inters_points_l, (0, 0, 255), 2)
cv.line(img, point_middle, first_inters_points_r, (0, 0, 255), 2)

cv.circle(img, point_b, 4, (255, 0, 0), -1)
cv.circle(img, point_a, 4, (255, 0, 0), -1)

cv.circle(img, point_middle, 4, (255, 0, 0), -1)
cv.imwrite('result.png', img)
