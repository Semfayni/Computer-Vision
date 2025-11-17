import pandas as pd #working with csv tables
import numpy as np #mathematics
import tensorflow as tf #ai
from tensorflow import keras #розширення
from tensorflow.keras import layers #РОЗШИРЕННЯ для розширення
from sklearn.preprocessing import LabelEncoder #текстові мітки в числа
import matplotlib.pyplot as plt #графічно виводити статистику

#1 ввели
df = pd.read_csv("data/figuresdz.csv")
df['area_perim_ratio'] = df['area'] / df['perimeter']
#2 зчитали
#print(df.head())

#3 name in numbers
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners', 'area_perim_ratio']]
y = df['label_enc']

#4 створення моделі
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape = (4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='softmax')
])

#compil
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#learning
history = model.fit(X, y, epochs = 500, verbose = 0)

#visualisation

plt.plot(history.history['loss'], label = 'Втрати')
plt.plot(history.history['accuracy'], label = 'Точність')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Process learning model')
plt.legend()
plt.show()

#test
test_area = 25
test_perimeter = 20
test = np.array([[test_area, test_perimeter, 0, test_area / test_perimeter]])
pred = model.predict(test, verbose=0)
print(f'Імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax(pred)])}')