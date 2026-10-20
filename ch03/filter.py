import pandas as pd

# read data in
faults_df = pd.read_csv("datasets_and_media/faults.csv")

# indexing
# return a series with the 'X_Minimum' column of the dataset
X_Minimum_series = faults_df['X_Minimum']
print("X_Minimum series")
print(X_Minimum_series)

# get 'X_Minimum' values as list
X_Minimum_list = faults_df['X_Minimum'].values.tolist()
print("X_Minimum List - First 10 values")
print(X_Minimum_list[0:10])

# loc
print("Entire X_Minimum column")
print(faults_df.loc[:, "X_Minimum"])
print("First X_Minimum value")
print(faults_df.loc[0, "X_Minimum"])
print("2nd thru 5th indexed X_Minimum values")
print(faults_df.loc[2:5, "X_Minimum"])
print("0th thru 3rd X_Minimum values")
print(faults_df.loc[:3, "X_Minimum"])
print("1939th thru last X_Minimum values")
print(faults_df.loc[1939:, "X_Minimum"])

print("0th thru 5th indexed X_Minimum and Pixels_Areas values")
print(faults_df.loc[0:5, ["X_Minimum", "Pixels_Areas"]])

# iloc
print("Row 1-2, Column 1-4")
print(faults_df.iloc[1:3, 1:5])
print("4th row, 0th column")
print(faults_df.iloc[4, 0])
print("4th row, all columns")
print(faults_df.iloc[4, :])

# select by condition
print("All rows where X_Minimum is above 500")
print(faults_df.loc[faults_df["X_Minimum"] > 500])
print("row count based on condition")
print(len(faults_df.loc[faults_df["X_Minimum"] > 500].index))

# create new dataframe from selection
print("All rows where X_Minimum is above 500 and Pixels_Areas is equal to 176")
sub_df = faults_df.loc[(faults_df["X_Minimum"] > 500) &
                       (faults_df["Pixels_Areas"] == 176)]
print(sub_df.head())
print("row count of sub_df")
print(len(sub_df.index))
