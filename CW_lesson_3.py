import cv2
import numpy as np

img = np.zeros((512, 400, 3), np.uint8)

#rgb = bgr
#all coloring
# img[:] = 95, 217, 24
#fragment coloring
# img[100:150, 200:280] = 95, 217, 24

cv2.rectangle(img, (100, 100), (200, 200), (95, 217, 24), 1)

cv2.line(img, (100, 100), (200, 200), (95, 217, 24), 1)
print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (95, 217, 24), 1)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (95, 217, 24), 1)

cv2.circle(img, (200, 200), 20, (95, 217, 24), 1)

cv2.putText(img, "Pugach Vsevolod", (200,300),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (95, 217, 24), 1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()