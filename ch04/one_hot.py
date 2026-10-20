import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
print("Before One-Hot Encoding:")
print(df["fault"])
encoder = OneHotEncoder()
encoder.fit(df[["fault"]])
print(encoder.categories_)
one_hot_encoded_outcomes = encoder.transform(df[["fault"]]).toarray()
print("After One-Hot Encoding:")
print(one_hot_encoded_outcomes)
