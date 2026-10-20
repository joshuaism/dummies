import pandas as pd
from sklearn.model_selection import train_test_split

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
print("Full Dataset Size:")
print(len(df.index))
train_set, test_set = train_test_split(df, test_size=0.1)
print("Train Set Size:")
print(len(train_set.index))
print(len(test_set.index))
