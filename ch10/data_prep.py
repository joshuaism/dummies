from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

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
