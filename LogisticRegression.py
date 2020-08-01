#This files aims to creating a logistic regression algorithm with scikit-learn.

import numpy as np
import pandas as pd
from _decimal import DecimalException
from sklearn.linear_model import LogisticRegression as lr

pd.options.display.max_columns = 99

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

df['male'] = df['Sex'] == 'male'

#Features
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values

#Target
y = df['Survived']
print(y)
#Definition of the model as a lr object:
model = lr()

#Building the model: fit creates the most appropriate line to split the classes
model.fit(X,y)

#printing the coefficients for knowledge:
print('\nModel: ' + '0 = ' + str(model.coef_) + ' + ' + str(model.intercept_))
print()

#Letting sklearn predict who will survived based on the dataframe:
y_pred = model.predict(X)
print(y_pred)

#Checking model's predictions
y_pred = model.predict(X)
print(y_pred)
print('The model got ' + str((y_pred == y).sum()) + ' predictions right.')

#Asking for a prediction based on arbitrary data:
if model.predict([[1, False, 31, 0, 2, 250.0]]) == 1:
    print('\nThe specified passenger survived.')
elif model.predict([[1, False, 31, 0, 2, 250.0]]) == 0:
    print('\nThe specified passenger did not survive.')
print()

#Calculating accuracy ratio (y.shape[0] counts the rows of the y dataframe):
# dumb way: print(str(round(((y_pred == y).sum() / y.shape[0]) * 100, 3)) + '% accuracy')
#clever way:
print(str(round(model.score(X, y) * 100, 3)) + '% accuracy')
