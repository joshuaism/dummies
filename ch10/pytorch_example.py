from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

# PyTorch Data Preparation and Model Implementation


class customIrisDataset(Dataset):
    def __init__(self, features, truth):
        self.features = features
        self.truth = truth

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        feature = self.features[idx]
        truth = self.truth[idx]
        return feature, truth


batch_size = 8

train_data = customIrisDataset(train_features, train_truth)
val_data = customIrisDataset(val_features, val_truth)
test_data = customIrisDataset(test_features, test_truth)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# PyTorch Neural Network


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_layers = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        output = self.model_layers(x)
        return output


model = NeuralNetwork()
# Get parameters to be the same type
model.double()
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)


def train(dataloader, model, loss_function, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(batch)
        pred = model(X)
        loss = loss_function(pred, y)
        # Backprop - the training step!
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_function):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_truth, in dataloader:
            predictions = model(batch_inputs)
            total_loss += loss_function(predictions, batch_truth).item()
            num_correct_pred = (predictions.argmax(
                1) == batch_truth.argmax(1)).type(torch.float).sum().item()
            total_correct += num_correct_pred
    overall_loss = total_loss/num_batches
    overall_accuracy = total_correct/num_samples
    return overall_loss, overall_accuracy


epochs = 50
training_loss_history = []
training_acc_history = []
val_loss_history = []
val_acc_history = []

for i in range(epochs):
    print(f"Epock {i}")
    train(train_dataloader, model, loss_function, optimizer)
    # Get training history by epoch
    train_loss, train_acc = test(train_dataloader, model, loss_function)
    training_loss_history.append(train_loss)
    training_acc_history.append(train_acc)
    # Get validation history by epoch
    val_loss, val_acc = test(val_dataloader, model, loss_function)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    print(f"Current Train Loss: {train_loss}, Accuracy: {train_acc}")
    print(f"Current Val Loss: {val_loss}, Accuracy: {val_acc}")

# Final test
loss, acc = test(test_dataloader, model, loss_function)
print(f"Final Test Loss: {loss}")
print(f"Final Test Accuracy: {acc}")

# Plot model accuracy
plt.plot(training_acc_history, label="accuracy")
plt.plot(val_acc_history, label="val_accuracy")
plt.title("Model Training and Val Accuracy over Time")
plt.legend()
plt.show()
plt.clf()

# Plot model loss
plt.plot(training_loss_history, label="loss")
plt.plot(val_loss_history, label="val_loss")
plt.title("Model Training and Val Loss over Time")
plt.legend()
plt.show()
plt.clf()
