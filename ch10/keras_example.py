from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

# Dataset Prep
# Load in Dataset Information
iris_full_dataset = load_iris()
iris_features_names = iris_full_dataset.feature_names
iris_label_classes = iris_full_dataset.target_names
iris_input_features = iris_full_dataset.data
iris_ground_truth = iris_full_dataset.target

# Print Dataset Information
print("Iris Features: ", iris_features_names)
print("Iris Label Classes", iris_label_classes)
print("First 10 instances")
print(iris_input_features[0:10])
print("First 10 class labels")
print(iris_ground_truth[0:10])

# Normalize  the Input Data
scaler = MinMaxScaler()
scaled_iris_input_features = scaler.fit_transform(iris_input_features)
print("After Normalizing Features")
print(scaled_iris_input_features[0:10])

# One Hot Encode the Ground Truth Data
one_hot = OneHotEncoder(sparse_output=False)
iris_ground_truth_one_hot = one_hot.fit_transform(
    iris_ground_truth.reshape(-1, 1))
print("After Converting to One Hot")
print(iris_ground_truth_one_hot.shape)
print(iris_ground_truth_one_hot[0:10])

train_features, test_features, train_truth, test_truth = train_test_split(
    scaled_iris_input_features, iris_ground_truth_one_hot, test_size=0.2)
train_features, val_features, train_truth, val_truth = train_test_split(
    train_features, train_truth, test_size=0.2)

# Keras Model Implementation
model = Sequential()
# 4 feature inputs
# some dense layers with ReLU activation
# 3 feature outputs
model.add(Input(shape=(4, )))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
print("Model Summary")
print(model.summary())

# stop training early if validation loss hasn't improved after three consecutive epochs
early_stop_callback = EarlyStopping(patience=3)
# train the model
history = model.fit(train_features, train_truth, batch_size=8, epochs=50, validation_data=(
    val_features, val_truth), callbacks=[early_stop_callback])

# test on unseen data
results = model.evaluate(test_features, test_truth)
print(f"Final Test loss: {results[0]}")
print(f"Final Test accuracy: {results[1]}")

# Plot model accuracy
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Model Training and Val Accuracy over Time")
plt.legend()
plt.show()
plt.clf()

# Plot model loss
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Model Training and Val Loss over Time")
plt.legend()
plt.show()
plt.clf()
