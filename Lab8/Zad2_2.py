import copy
import numpy as np
import cv2 as cv
from scipy import spatial
from math import isclose
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


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

img = cv.imread('Lab8/s1.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_blue = np.array([0, 60, 180])
upper_blue = np.array([20, 140, 255])

mask = cv.inRange(hsv, lower_blue, upper_blue)

mask2 = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4)), iterations=2)
mask2 = cv.morphologyEx(mask2, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8)), iterations=3)
cv.imwrite("Lab8/mask.png", mask2)
img2 = img.copy()

for i in range(3):
    img2[:, :, i] *= mask2
cv.imwrite("Lab8/s2.png", img2)
img2_g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

aug = iaa.pillike.EnhanceSharpness(factor=2.0)
img3 = aug.augment_image(img2)
aug = iaa.MedianBlur(k=3)
img3 = aug.augment_image(img3)
img4 = cv.Canny(img3, 50, 240)
plt.imshow(img4)

cv.imwrite("Lab8/s2.png", img3)
contours, hierarchy = cv.findContours(copy.deepcopy(img4), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
contours_f = [contours[i].reshape((-1, 2)) for i in [0, 5, 13]]
points = np.concatenate(contours_f)
center = points.mean(axis=0)
points_angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
img5 = copy.copy(img2)
img6 = cv.fillConvexPoly(img5, points=points[points_angles.argsort()], color=(255, 255, 255))
img7 = cv.cvtColor(img6, cv.COLOR_BGR2GRAY)
img7[img7 < 255] = 0
plt.imshow(img7)
points = points[points_angles.argsort()]
mask = img7.copy()
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
