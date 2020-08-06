import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

pd.options.display.max_columns = 99

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

#Splitting the data in a training dataset and a testing dataset, to 
#better evaluate the model's performance on new data
#Random_state helps avoiding getting a different result every time
#the model is run due to the randomness in train_test_split. 
#Random_state makes sense when you actually need to see the same result
#multiple times, it would be better to allow randomness when
#testing the model, since the metrics change between runs too.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1)

print('The whole dataset has {0} features and {1} observations.'.format(X.shape[1], y.shape[0]))
print('The training dataset has still {0} features, but {1} observations.'.format(X_train.shape[1], y_train.shape[0]))
print('The testing dataset has still {0} features, but {1} observations.'.format(X_test.shape[1], y_test.shape[0]))
print()
#All of the model building will be done with the training set and
#all of the evaluation will be done with the test set

model = lr()
model.fit(X_train, y_train)

print('Predictions: \n', y_pred := model.predict(X_test))
print()
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision : ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1: ', f1_score(y_test, y_pred))
print()

