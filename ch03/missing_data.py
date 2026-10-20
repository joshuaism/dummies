import pandas as pd

# read teh dataset into a variable
faults_df = pd.read_csv("datasets_and_media/faults.csv")

# find and handle NaNs
print(faults_df.isna().value_counts())

# insert NaN value
print("initial value")
print(faults_df.loc[0, 'X_Minimum'])
new_df = faults_df
new_df.loc[0, 'X_Minimum'] = None
print("updated value")
print(new_df.loc[0, 'X_Minimum'])

# Drop na value
print("initial row count")
print(len(new_df.index))
cleaned_df = new_df.dropna()
print("DataFrame after droping NaN values - note index")
print(cleaned_df.head())
# Reset Index
cleaned_df.reset_index(inplace=True)
print("updated row count")
print(len(cleaned_df.index))
print("Value at previously-NaN index")
print(cleaned_df.loc[0, "X_Minimum"])
print("DataFDrame after resetting NaN values - note index")
print(cleaned_df.head())
