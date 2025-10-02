import cv2
import numpy as np

img = cv2.imread('images/imgpr.jpg')
qrcode = cv2.imread('images/qrcode.jpg')
img_resized = cv2.resize(img, (130, 130))
qrcode_resized = cv2.resize(qrcode, (100, 100))
bg = np.zeros((400, 600, 3), np.uint8)
bg[:] = (194, 229, 242)
cv2.rectangle(bg, (5, 5), (595, 395), (100, 197, 232), 2)
bg[30:160, 30:160] = img_resized
bg[270:370, 470:570] = qrcode_resized
cv2.putText(bg, "Pugach Vsevolod", (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(bg, "Computer Vision Student", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (69, 69, 69), 2)
cv2.putText(bg, "Email: l901752@gmail.com", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (112, 0, 6), 1)
cv2.putText(bg, "Phone: +380956740158", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (112, 0, 6), 1)
cv2.putText(bg, "30/10/2009", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (112, 0, 6), 1)
cv2.putText(bg, "OpenCV Business Card", (110, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imshow('BUSINESS CARD', bg)
cv2.imwrite("business_card.png", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()