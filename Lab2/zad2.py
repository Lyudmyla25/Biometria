import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal as sg

img = cv.imread('super_big_eagle.jpg')


def filtration(img, matrix):
    if img.ndim == 2:
        return sg.convolve2d(img, matrix[:, ::-1], "valid").astype(int)
    if img.ndim == 3:
        res_lst = [sg.convolve2d(img[:, :, x], matrix[:, ::-1], "valid") for x in range(img.shape[2])]
        return np.rollaxis(np.array(res_lst), 0, 3).astype(int)


filters = dict({"rozmycie": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0,
                "gaussian_blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0,
                "wyostrzenie": np.array([[0, -2, 0], [-2, 11, -2], [0, -2, 0]]),
                "wykrywanie_krawedzi": np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])})


for title, matrix in filters.items():
    print(title)
    plt.imshow(filtration(img, matrix))
    plt.title(title)
    cv.imwrite(title + ".jpg", filtration(img, matrix))
    plt.show()

