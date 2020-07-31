import numpy as np
import pandas as pd
#the following line of code sets the max number of displayed columns to 99,
#otherwise with the default settings they wouldn't all be displayed.
pd.options.display.max_columns = 99

data = [15, 16, 18, 19, 22, 24, 29, 30, 34]

print("mean:", np.mean(data))
print("median:", np.median(data))
print("50th percentile (median):", np.percentile(data, 50))
print("25th percentile:", np.percentile(data, 25))
print("75th percentile:", np.percentile(data, 75))
print("standard deviation:", round(np.std(data), 2))
print("variance:", round(np.var(data), 2))

df = pd.read_csv('titanic.csv')
print(df.head())
print(df.describe())