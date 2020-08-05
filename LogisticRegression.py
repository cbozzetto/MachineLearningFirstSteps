#This files aims to creating a logistic regression algorithm with scikit-learn.

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
print(y_pred) #or, with walrus: print(y_pred := model.predict(X))

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
print()
#Other measuers of model performance
print('Accuracy: ', accuracy_score(y, y_pred))
print('Precision : ', precision_score(y, y_pred))
print('Recall: ', recall_score(y, y_pred))
print('F1: ', f1_score(y, y_pred))
print()
print('Confusion matrix (first col = predicted negatives; second col = predicted positives);\
\nfirst row = actual negatives; second row = actual positives \n')
print(confusion_matrix(y, y_pred))
print()
print('Precision = 239/309 = ', 239/309)
print('Recall = 239/342 = ', 239/342)
print('F1 = (harmonic mean) = 2 * (Precision * Recall) / (Precision + Recall) = \
', 2 * ((239/309)*(239/342))/((239/309)+(239/342)))
print()
