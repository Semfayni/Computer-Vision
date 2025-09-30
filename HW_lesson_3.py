import cv2
import numpy as np

img = cv2.imread('images/img4.jpg')
resized = cv2.resize(img, (800, 400))

cv2.rectangle(resized, (360, 100), (520, 300), (95, 217, 24), 2)
cv2.putText(resized, "Pugach Vsevolod", (370, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (95, 217, 24), 1)

cv2.imshow('resized', resized)
cv2.waitKey(0)