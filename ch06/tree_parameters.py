import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

faults = ['Pastry', 'Z_Scratch', 'K_Scatch',
          'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']


def read_in_dataset(faults):
    df = pd.read_csv("datasets_and_media/faults.csv")
    df['fault'] = 0
    # create categorical variable for each fault
    for i in range(0, len(faults)):
        true_fault_indexes = df.loc[df[faults[i]] == 1].index.tolist()
        df.loc[true_fault_indexes, "fault"] = i+1
    return df


df = read_in_dataset(faults)
drop_features = ["fault"] + faults
# features is the table minus fault columns (which are outcomes) which our model should base predictions on
features = df.drop(drop_features, axis=1)
# outcomes is the "catagorical" fault column (based on the bool values in ["fault"] columns) our model should predict
outcomes = df["fault"]
# create the training and test set
training_features, test_features, training_outcomes, test_outcomes = train_test_split(
    features, outcomes, test_size=0.1)

model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, min_samples_leaf=4)

model.fit(training_features, training_outcomes)
test_accuracy_score = model.score(test_features, test_outcomes)
training_accuracy_score = model.score(training_features, training_outcomes)
print(f"Training accuracy {training_accuracy_score}")
print(f"Test accuracy {test_accuracy_score}")
plot_tree(model, fontsize=8, filled=True)
plt.show()
plt.clf()
