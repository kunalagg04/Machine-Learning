# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:43:02 2019
@author: Kunal Aggarwal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing dataset using pandas
dataset = pd.read_csv('Data.csv')
#creating independent variable vector 
X = dataset.iloc[:,:-1].values
#creating dependent variable vector 
Y = dataset.iloc[:,-1].values


#handling missing data 
#we put an average of other values in place of missing data
#creating object of class imputer
#missing_values will recognise missing values and strategy= mean will put mean of other corresponding values
#axis : heps to choose if we want to take mean of column(0) or row(1) 
imputer = Imputer(missing_values="NaN", strategy="mean",axis=0)
#fit will fit imputer object to our matrix X
imputer = imputer.fit(X[:,1:3])
#transform will replace x TO NEW X with mean in place of missing data 
X[:, 1:3]=imputer.transform(X[:, 1:3])


#handling categorial data 
#country name and YES/NO are categorial data . They need to be replaced by numbers
labelencoder_X = LabelEncoder()
#THIS WILL give 0,1,2 values to countries   
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#providing integer value to countries will make Ml model think different countries have different weightage !
#so we create dummy variables and ceate three columns in place of country column
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
#FOR HANDLING depedent variable categorical data we need not call onehotencoder coz ML knows ki ye dependent variable hai to Categorical data hi hoga
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#breakind data to training and test data
#use random state parameter to shuffle data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , random_state = 0)

#feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)
#featre scaling of dummy variables depends on context 
#NO need to apply feature scaling on Y for classification , apply for regression 
