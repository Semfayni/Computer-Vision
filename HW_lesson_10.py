import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#1 СТВОРЮЄМО ФУНКЦІЮ ДЛЯ ГЕНЕРАЦІЇ ПРОСТИХ ФІ

#2 ФОРМУЄМО НАБОРИ ДАНИХ

#список ознак
X = []
#список міток
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "orange": (0, 106, 255),
    "cyan": (255, 255, 0),
    "black": (0, 0, 0)
}

for color_name, bgr in colors.items():
    for i in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)


#x-train ознак для навчання х тест для перевирк у трейн для навчаннчя у тест для перевирки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#4 НАВЧАЄМО МОДЕЛЬ
model = KNeighborsClassifier(n_neighbors=3)#бажано ставити НЕПАРНІ числа
model.fit(X_train, y_train)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1111:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]

            mean_color = cv2.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1, -1)

            label = model.predict(mean_color)[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            accuracy = model.score(X_test, y_test)
            cv2.putText(frame, label.upper(), (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'{round(accuracy * 100, 2)}%', (x+105, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()