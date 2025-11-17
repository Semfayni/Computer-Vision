import pandas as pd #working with csv tables
import numpy as np #mathematics
import tensorflow as tf #ai
from tensorflow import keras #розширення
from tensorflow.keras import layers #РОЗШИРЕННЯ для розширення
from sklearn.preprocessing import LabelEncoder #текстові мітки в числа
import matplotlib.pyplot as plt #графічно виводити статистику

#1 ввели
df = pd.read_csv("data/figures.csv")

#2 зчитали
#print(df.head())

#3 name in numbers
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

#4 створення моделі
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape = (3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(8, activation='softmax')
])

#compil
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#learning
history = model.fit(X, y, epochs = 300, verbose = 0)

#visualisation

plt.plot(history.history['loss'], label = 'Втрати')
plt.plot(history.history['accuracy'], label = 'Точність')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Process learning model')
plt.legend()
plt.show()

#test
test = np.array([[25, 20, 0]])
pred = model.predict(test)
print(f'Імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax(pred)])}')