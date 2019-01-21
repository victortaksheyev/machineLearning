#SVR
"""
Created on Sun Jan 20 18:28:25 2019

@author: victortaksheyev
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


# Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Fitting the SVR Model to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting New Result
y_pred = sc_y.inverse_transform(regressor.predict(X_test))

# Checking accuracy

R2=r2_score(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
print("R squared value:",R2, '\n' + "Mean squared error:",MSE)


