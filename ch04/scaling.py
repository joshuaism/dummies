import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
print("Before MinMaxScaling:")
print(df["X_Minimum"])
# scale the X_Minimum values between 0 and 1
scaler = MinMaxScaler()
scaled_x_min = scaler.fit_transform(df[["X_Minimum"]])
print("After MinMaxScaling:")
print(scaled_x_min)
