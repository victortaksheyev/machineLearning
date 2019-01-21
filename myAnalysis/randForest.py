#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:36:56 2019

@author: victortaksheyev
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 3)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Accuracy
R2=r2_score(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
print("R squared value:",R2, '\n' + "Mean squared error:",MSE)

Xnew = dataset.drop(['Chance of Admit ', 'Serial No.'], axis=1)

regressor.feature_importances_

'''
feature_importance = pd.DataFrame(sorted(zip(regressor.feature_importances_, Xnew.columns)), columns=['Importance','Variable'])
plt.figure()
sns.barplot(x="Importance", y="Variable", data=feature_importance, palette="rocket")
plt.title('Variable Importance')
plt.tight_layout()

'''
y_pred = y_pred.reshape(80, 1)

comb = np.concatenate((y_test, y_pred), axis=1)

comb = pd.DataFrame(data=comb[:80, :2],    # values 
             index=None,    # 1st column as index
             columns=['Real Chance of Admission', 'Predicted Chance of Admission'])  # 1st row as the column names

comb[:80, :2]


plt.bar(Xnew.columns, regressor.feature_importances_, width=0.8, bottom=None, align='center', data=None, color=(	71/255, 106/255, 163/255))
plt.title('Importance of Variables')
plt.ylabel('Importance')
plt.xlabel('Variables')
plt.show()