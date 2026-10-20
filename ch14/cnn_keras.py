from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import cv2 as cv
import numpy as np
import random

# Dataset Pret
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training and Test Set X Shape")
print(x_train.shape)
print(x_test.shape)
print("Training and Test Set Y Shape")
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(y_train.shape)
print(y_test.shape)

# Model Prep
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(16, 3, activation="relu"))
model.add(MaxPool2D(pool_size=(4, 4)))
model.add(Conv2D(8, 3, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
print("Model Summary")
print(model.summary())

# Model Training
history = model.fit(x_train, y_train, batch_size=16, epochs=5)

# Model Evaluation
results = model.evaluate(x_test, y_test)
print(f"Final Test loss: {results[0]}")
print(f"Final Test accuracy: {results[1]}")

# Spot Check the Model
key = ord("n")
random_index = random.randint(0, 100)
while key != ord("s") and key != ord("S"):
    pic_to_predict = np.expand_dims(x_test[random_index], axis=0)
    prediction = model.predict(pic_to_predict)
    digit_prediction = np.argmax(prediction)
    img = cv.resize(x_test[random_index], (280, 280))
    cv.imshow(f"Prediction: {digit_prediction}", img)
    key = cv.waitKey()
    if key == ord("r"):
        random_index = random.randint(0, 100)
cv.destroyAllWindows()
