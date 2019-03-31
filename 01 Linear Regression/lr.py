# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 02:42:19 2019

@author: Kunal Aggarwal
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=1/3 , random_state = 0)

'''feature scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)'''


#Fitting linear regression to training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#fitting regressor to training set made it learn corelations between dependent & independent variables 


#predicting test set results
y_pred = regressor.predict(X_test)


#plotting graph of training set 
#scatter means point plot honge
plt.scatter(X_train , Y_train , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.title('Salary vs Experience(TRAINING SET)') 
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#plotting graph of test set 
plt.scatter(X_test , Y_test , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.title('Salary vs Experience(TEST SET)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
