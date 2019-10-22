# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

1
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pd.read_csv('RedWineQuality.csv')
print(dataset.head())
print(dataset.describe())

x = dataset.drop('quality', axis=1)
y = dataset.quality

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, stratify=y)

x_train_scaled = preprocessing.scale(x_train)
print (x_train_scaled)