# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:31:45 2019

@author: Kunal Aggarwal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


labelencoder_X = LabelEncoder()  
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
#removing first column 
X = X[:, 1:]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , random_state = 0)


#Fitting multiple linear regression to  training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

#predicting test set result
Y_predict = regressor.predict(X_test)


