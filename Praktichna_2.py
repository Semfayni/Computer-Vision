import cv2
import numpy as np

img = cv2.imread("images/pr2.jpg")
img = cv2.resize(img, (800, 600))
img_copy = img.copy()

img = cv2.GaussianBlur(img_copy, (3, 3), 0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([0, 16, 0])
upper_yellow = np.array([38, 255, 255])

lower_blue = np.array([108, 40, 0])
upper_blue = np.array([179, 255, 132])

lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 104, 147])

lower_green = np.array([40, 54, 78])
upper_green = np.array([51, 255, 191])

mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_black = cv2.inRange(img, lower_black, upper_black)
mask_green = cv2.inRange(img, lower_green, upper_green)

mask_total = cv2.bitwise_or(mask_yellow, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_black)
mask_total = cv2.bitwise_or(mask_total, mask_green)
img = cv2.bitwise_and(img, img, mask=mask_total)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
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
        elif len(approx) >= 8:
            shape = "Oval"
        else:
            shape = "Parrot"
        if int(area) == 73772:
            color = "yellow"
        elif int(area) == 10736:
            color = "blue"
        elif int(area) == 10687:
            color = "black"
        elif int(area) == 9693:
            color = "green"
        else:
            color = "another"
        cx1 = cx - w/2
        cy1 = cy - h/2
        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'CX: {cx1}, CY: {cy1}', (x - 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img_copy, f'Shape: {shape}', (x-70, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img_copy, f'S: {int(area)}', (x+40, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, f'Color: {color}', (x + 120, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow("Original", img)
cv2.imshow("Gray", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()