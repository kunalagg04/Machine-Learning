# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:28:40 2019

@author: Kunal Aggarwal
"""
#Aim : We need to determine salary of LEVEL 6.5 worker 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


#In Position_Salaries data , Position column is equivalent to Grade column. Hnece, we need not use position column
dataset = pd.read_csv('Position_Salaries.csv')
#we want independent variable to be a matrix and not an error
X = dataset.iloc[:, 1:2].values  
Y = dataset.iloc[:, 2].values

#no splitting of data coz of small dataset and we need accurate prediction of salary !

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
#poly reg will create a new matrix of features X conatining powers of X
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#in x_poly , we have Three columns , x0 is created by library itself. Now we apply linear regression to x_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , Y)
 
#Visualising linear regression results
plt.scatter(X,Y, color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary') 
plt.show()

#visualising polynomial regression results 
#Note: X_grid has been created and used because original graph has kinda staright lines between different lines 
#So we used X_grid to have smoothness in curve 

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#predicting new result with LR 
lin_reg.predict(6.5)

#predicting new result with PLR
lin_reg_2.predict(poly_reg.fit_transform(6.5))
