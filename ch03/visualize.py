import pandas as pd
import matplotlib.pyplot as plt

# read teh dataset into a variable
faults_df = pd.read_csv("datasets_and_media/faults.csv")

# Plot - 1 Variable
plt.plot(faults_df["X_Minimum"])
plt.show()
plt.clf()

# Scatter Plot - 2 Variables
plt.scatter(faults_df["X_Minimum"], faults_df["X_Maximum"])
plt.show()
plt.clf()

# Histogram of Counts
plt.hist(faults_df["Pastry"])
plt.show()
plt.clf()
