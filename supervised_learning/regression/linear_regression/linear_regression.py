
pwd
#SIMPLE LINEAR REGRESSION

#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('ggplot')

#IMPORTING THE DATASET
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#SPLITTING THE DATASET INTO THE TRAINING SET AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 1/3, random_state=0)

#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

#VISUALIZING THE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color= "green")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience (Training set)')
plt.xlable('Years of Experience')
plt.ylabel('Salary')
plt.show()

#

