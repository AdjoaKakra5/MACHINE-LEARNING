#SUPPORT VECTOR MACHINE

#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import style
style.use ('ggplot')

#IMPORTING THE DATASET
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

#SPLITTING THE DATASET INTO THE TRAINING SET AND THE TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#FITTING SVM TO THE TRAINING SET
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
