import cv2
import numpy as np
image1 = cv2.imread('images/img1.jpg')
resized_image1 = cv2.resize(image1, (300, 300))
gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray_image1, 100, 150)
cv2.imshow('edges1', edges1)

image2 = cv2.imread('images/img21.jpg')
resized_image2 = cv2.resize(image2, (600, 300))
gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(gray_image2, 205, 212)

cv2.imshow('edges2', edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()