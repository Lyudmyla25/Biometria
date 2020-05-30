import numpy as np
import cv2
import skimage
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square


def getTerminationBifurcation(img, mask):
    img = img == 255
    (rows, cols) = img.shape
    minutiaeTerm = np.zeros(img.shape)
    minutiaeBif = np.zeros(img.shape)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (img[i][j] == 1):
                block = img[i - 1:i + 2, j - 1:j + 2]
                block_val = np.sum(block)
                if (block_val == 2):
                    minutiaeTerm[i, j] = 1
                elif (block_val == 4):
                    minutiaeBif[i, j] = 1

    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(5))
    minutiaeTerm = np.uint8(mask) * minutiaeTerm
    return (minutiaeTerm, minutiaeBif)


def removeSpuriousMinutiae(minutiaeList, img, thresh):
    img = img * 0
    SpuriousMin = []
    numPoints = len(minutiaeList)
    D = np.zeros((numPoints, numPoints))
    for i in range(1, numPoints):
        for j in range(0, i):
            (X1, Y1) = minutiaeList[i]['centroid']
            (X2, Y2) = minutiaeList[j]['centroid']

            dist = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2);
            D[i][j] = dist
            if (dist < thresh):
                SpuriousMin.append(i)
                SpuriousMin.append(j)

    SpuriousMin = np.unique(SpuriousMin)
    for i in range(0, numPoints):
        if (not i in SpuriousMin):
            (X, Y) = np.int16(minutiaeList[i]['centroid']);
            img[X, Y] = 1

    img = np.uint8(img)
    return (img)


def makeMinucja(img):
    img = np.uint8(img > 128)
    skel = skimage.morphology.skeletonize(img)
    skel = np.uint8(skel) * 255
    mask = img * 255
    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)

    minutiaeTerm = skimage.measure.label(minutiaeTerm, 8)
    RP = skimage.measure.regionprops(minutiaeTerm)
    minutiaeTerm = removeSpuriousMinutiae(RP, np.uint8(img), 10)

    BifLabel = skimage.measure.label(minutiaeBif, 8)
    TermLabel = skimage.measure.label(minutiaeTerm, 8)

    minutiaeBif = minutiaeBif * 0
    minutiaeTerm = minutiaeTerm * 0

    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel
    DispImg[:, :, 1] = skel
    DispImg[:, :, 2] = skel
    RP = skimage.measure.regionprops(BifLabel)
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

    RP = skimage.measure.regionprops(TermLabel)
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))
    return DispImg


img = cv2.imread('super_big_fp.png', 0)[:, :-50]
m, n = img.shape
images = dict()
for i in range(2):
    for j in range(3):
        key = 3 * i + j
        images.update({key: img[i * m // 2:(i + 1) * m // 2, j * n // 3:(j + 1) * n // 3]})

for i in range(6):
    img = cv2.bitwise_not(images[i])
    result_img = makeMinucja(img)
    cv2.imwrite(f'result{i+1}.png', result_img)

