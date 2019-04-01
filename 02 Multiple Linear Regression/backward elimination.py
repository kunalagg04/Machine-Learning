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

#building model with backward elimination
import statsmodels.formula.api as sm
#adding x0 column to X matri
# Doing this will add x0 to end => X = np.append(arr = X, value = np.ones(50,1).astype(int), axis = 1 )
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1 )
X_opt = X[: , [0,1,2,3,4,5]] 
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
#x2 has highest p value . Hence we removeit nd fit model again.
X_opt = X[: , [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
#now  NEW x1 has highest p value . Hence we remove it as well.
X_opt = X[: , [0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
#now new x2 has highest p value . It's index is 4 according to original X
X_opt = X[: , [0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
