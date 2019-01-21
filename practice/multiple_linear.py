# multiple linear regression

#data preprocessing ------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
#must take one dummy variable away for multiple linear regression with n dummy variables ...
# ...such that n > 1
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#prediciting the Test set results
y_pred = regressor.predict(X_test)

#Optimizing model with Backward Elimination
import statsmodels.formula.api as sm
#adding matrix X to a column of 1s
X = np.append(arr=np.ones((50, 1)).astype(int), values = X, axis=1)

"""
#Doesn't work because p value changes with every backward elimination performed
#... and the model is only fitted once instead of every time an elimination is performed
def backwardElim(x, y, sl):
    toDel = []
    numVars = len(X_opt[1])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        pVals = regressor_OLS.pvalues.astype(float)
    for j in range(0, numVars):
        if pVals[j] > float(sl):
            toDel.append(j)
    
    x = np.delete(x, toDel, 1)
    return x
          

#matrix of independent variables that have high impact on profit
X_opt = X[:, [0,1,2,3,4,5]]

X_new = backwardElim(X_opt, y, 0.95)
"""

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxP = max(regressor_OLS.pvalues).astype(float)
        if maxP > sl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxP):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)