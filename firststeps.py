import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
#head displays only the first 5 observations
print(df.head())
print(df.describe())

#the following code returns a Pandas series
cols = df['Survived']
print(cols)
print(cols.head())
print(cols.describe())

#the following code, with double brackets, creates a brand new Pandas dataframe comprised of the columns Survived
colsdf = df[['Survived']]
print(colsdf.head())
print(colsdf.describe())

#if we wanted to create a smaller dataframe starting from the titanic.csv file:
newdf = df[['Survived', 'Age', 'Fare']]
print(newdf.head())
print(newdf.describe())
#so: single brackets=one column; double brackets=multiple columns

#creating a new column
df['male'] = df['Sex'] == 'male'
newrdf = df[['Survived', 'Age', 'male']]
print(newrdf.head())
print(newrdf.describe())
#describe won't display stats for booleans

#now we convert a Pands series into a Numpy array, in order to do calculations:
print('\nList of ages:\n')
print(df['Age'].values)
print('\n')
#this would do the same thing:
print(np.array(df['Age']))

#multi-dimensional array
print('\n2-dim array:\n')
print(df[['Pclass', 'Fare', 'Age']].values)
print('The size of the array is [rows, cols]:')
print(df[['Pclass', 'Fare', 'Age']].values.shape)
print('The size of the main dataframe is [rows, cols]:')
print(df.values.shape)

#odd syntax to select just one column from an array (does not apply to rows)
arr = df[['Pclass', 'Fare', 'Age']].values
#displays the Age column
print(arr[:,2])

#what if we wanted data for not-underage-people only? We need to create a mask.
mask = arr[:,2] > 18
print(arr[mask])
print('\n')
#or, inline:
print(arr[arr[:,2] > 18])

#how many children were on the Titanic?
survage = df[['Survived', 'Age']].values
child = survage[:,1] < 18
deadchild = (survage[:,1] < 18) & (survage[:,0] == 0)
print('On the Titanic there were ' + str(child.sum()) + ' children, ' + str(deadchild.sum()) + ' of which died.')

#Let's plot, babyy!
plt.scatter(df['Age'], df['Fare'], c=df['Survived'])
plt.xlabel('Age')
plt.ylabel('Fare')
died_patch = mpatches.Patch(color = 'purple', label = 'Died')
surv_patch = mpatches.Patch(color = 'yellow', label = 'Survived')
plt.legend(handles = [died_patch, surv_patch])
plt.show()
