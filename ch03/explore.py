import pandas as pd

# read teh dataset into a variable
faults_df = pd.read_csv("datasets_and_media/faults.csv")

# get an overview
print("First 5 rows of faults_df")
print(faults_df.head())

# print all columns
print("Columns Index")
print(faults_df.columns)

# print all columns as a list
print("Columns as a list")
print(list(faults_df.columns))

# describe the data (basic statistical analysis)
print("Describe the data")
print(faults_df.describe())
