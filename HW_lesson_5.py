import cv2
import numpy as np

img = cv2.imread("images/shapes.jpg")
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#min and max поріг
lower = np.array([0, 0, 0])
upper = np.array([179, 255, 115])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        #Допомагає відрізняти співвідношення сторін
        aspect_ratio = round(w / h, 2)
        #Міра округлості об'єкта
        compactness = round((4 * np.pi * area / (perimeter ** 2)), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) == 10:
            shape = "Star"
        elif len(approx) >= 8:
            shape = "Oval"
        else:
            shape = "Another"

        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'Shape: {shape}', (x-100, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img_copy, f'S: {int(area)}, P:{int(perimeter)}', (x+50, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, f'AR: {aspect_ratio}, C:{compactness}', (x+250, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()