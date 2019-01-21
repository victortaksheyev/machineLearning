# multiple linear regression

#data preprocessing ------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 8].values

#Optimizing model with Backward Elimination
import statsmodels.formula.api as sm
#adding matrix X to a column of 1s
X = np.append(arr=np.ones((400, 1)).astype(int), values = X, axis=1)

X_opt = X[:, [0, 7]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred1 = regressor.predict(X_test)

R2=r2_score(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
print("R squared value:",R2, '\n' + "Mean squared error:",MSE)

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
X_opt = X[:, [0,1,2,3,4,5,6,7]]

X_new = backwardElim(X_opt, y, 0.05)

"""
import statsmodels.formula.api as sm
def backwardElim(x, y, sigl):
    numVars = len(x[1])
    for i in range(numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxP = max(regressor_OLS.pvalues).astype(float)
        if maxP > sigl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxP):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled = backwardElim(X_opt, y, SL)
"""

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred2 = regressor.predict(X_test)

R2=r2_score(y_test,y_pred2)
MSE=mean_squared_error(y_test,y_pred2)
print("R squared value:",R2, '\n' + "Mean squared error:",MSE)