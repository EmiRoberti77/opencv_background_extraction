import cv2
import numpy as np

img1 = cv2.imread("images/frame_71.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/frame_164.jpg", cv2.IMREAD_GRAYSCALE)

diff = cv2.absdiff(img1, img2)

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Difference", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
