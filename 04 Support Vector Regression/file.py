# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 00:27:37 2019

@author: Kunal Aggarwal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler



#In Position_Salaries data , Position column is equivalent to Grade column. Hence, we need not use position column
dataset = pd.read_csv('Position_Salaries.csv')
#we want independent variable to be a matrix and not an error
X = dataset.iloc[:, 1:2].values  
Y = dataset.iloc[:, -1].values
#convert Y to 2D array 
Y = Y.reshape(-1, 1)


#no splitting of data coz of small dataset and we need accurate prediction of salary !

#feature scaling 
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#fitting SVR to dataset
from sklearn.svm import SVR
#we can give linear,polynommial,sigmoid,rbf as argument . But we know we are not using linear right now.
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

#predicting result at 6.5 ! Here since we applied feature scaling manually we need to apply it here as well!
#[[6.5]] to convert 6.5 to a 2D array
#inverse_transform bcoz output will be on featured scale . hence we need to view resut in normal scale.
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising linear regression results
plt.scatter(X,Y, color = 'red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary') 
plt.show()

