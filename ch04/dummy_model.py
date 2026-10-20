import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

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
train_set, test_set = train_test_split(df, test_size=0.1)
train_features = train_set.drop(columns=faults+["fault"])
test_features = test_set.drop(columns=faults+["fault"])
train_outcomes = train_set["fault"]
test_outcomes = test_set["fault"]
model = DummyClassifier(strategy="most_frequent")
model.fit(train_features, train_outcomes)
avg_accuracy = model.score(test_features, test_outcomes)
print("Average model accuracy")
print(avg_accuracy)
predictions = model.predict(test_features)
print("Predictions")
print(predictions)
