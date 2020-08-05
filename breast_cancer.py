from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as lr
import pandas as pd


pd.options.display.max_columns = 99
#This data set is built in in sklearn, no need to read it as csv
cancer = load_breast_cancer()

#cancer object is now a sort of "dictionary"
print(cancer.keys())

print(cancer.DESCR)

#pull the data form the cancer obj to pandas
#we notice that the 'data' key is a 2-dim numpy array 
print(cdata := cancer.data)
print()
print(cdata.shape)
#we need to name the columns of the pandas dataframe: the names are stored in the 'feature_names' key
print()
print(ccols := cancer.feature_names)
print()
print(ccols.shape)
print()
#now we can create the pandas dataframe
df = pd.DataFrame(cdata, columns = ccols)
print(df.head())
print()

#we now need the target for our regression: the target values are stored in the 'target' key:
print(ctarg := cancer.target) #or cancer['target']
print()
print(ctarg.shape)
print()
#but we don't know whether benign is 0 or 1, let's check:
print('0 stands for ' + str(cancer.target_names[0])) #or cancer['target_names'][0]
print()
print('This dataset contains ' + str((ctarg == 1).sum()) + ' non-cancerous bodies and \
' + str((ctarg == 0).sum()) + ' cancerous bodies, for a total of \
' + str((ctarg == 1).sum() + (ctarg == 0).sum()) + ' observations.')
print()

#Let's add the target to the dataframe:
df['Malignant'] = ctarg == 0
print(df.head())
print()

#building the logistic regression model:
X = df[ccols].values #coefficients
y = df.Malignant.values #binary target variable value
model = lr(solver = 'liblinear') #creation of the model
model.fit(X,y) #computation of the best line that divides the classes Malignant and Benign
print('Model: 0 = ' + str(model.coef_) + ' + ' + str(model.intercept_))
print()

#using the model to predict entity of tumors:
print(pred := model.predict(X))
print()
print('According to the model, ' + str((pred == True).sum()) + ' bodies are Malignant \
and ' + str((pred == False).sum()) + ' bodies are benign, for a total of \
 ' + str((pred == True).sum() + (pred == False).sum()) + ' predictions.')
print()
print('This dataset contains ' + str((ctarg == 0).sum()) + ' Malignant bodies and \
' + str((ctarg == 1).sum()) + ' benign bodies, for a total of \
' + str((ctarg == 1).sum() + (ctarg == 0).sum()) + ' observations.')
print()
print('Accuracy: ' + str(round(model.score(X, y), 3) * 100) + '%')

