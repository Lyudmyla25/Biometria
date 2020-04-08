import cv2
from matplotlib import pyplot as plt


def add_images(img1, img2, w=0.5):
    return (w * img1 + (1 - w) * img2).astype("uint8")


img1 = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab3/stp1.png')
img2 = cv2.imread('/Users/I533569/PycharmProjects/Biometria/Lab3/stp2.png')
sum_img = add_images(img1, img2)

img1_median = cv2.medianBlur(img1, 3)
img2_median = cv2.medianBlur(img2, 3)

sum_img_median = cv2.medianBlur(sum_img, 3)
sum_of_median = add_images(img1_median, img2_median)

plt.imshow(sum_img_median)
plt.title('Median(A+B)')
plt.savefig("Lab3/Median(A+B).png")
plt.show()

plt.imshow(sum_of_median)
plt.title('Median(A)+Median(B)')
plt.savefig("Lab3/Median(A)+Median(B).png")
plt.show()

plt.hist(sum_img_median.ravel(), 256, [0, 256])
plt.title('Histogram Median(A+B)')
plt.show()


plt.hist(sum_of_median.ravel(), 256, [0, 256])
plt.title('Histogram Median(A)+Median(B)')
plt.show()
