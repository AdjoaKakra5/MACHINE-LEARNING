#K MEANS CLUSTERING 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style  
style.use('ggplot')

#IMPORTING THE DATASET

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#y = dataset.iloc[:, 3].values

#SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET

