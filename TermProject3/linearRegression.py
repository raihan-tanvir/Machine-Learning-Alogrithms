# Import the necessary libraries
import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pandas.read_csv('salaryData.csv')

#Explore the dataset
print(dataset.shape) # number of rows and columns
print(dataset.head(5)) # display first five rows of the dataset

# Differentiate attribute and target columns
x = dataset['YearsExperience'].values
y = dataset['Salary'].values

X = x.reshape(len(x),1)
Y = y.reshape(len(y),1)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 2/3)

linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)

# Predicting the test set results
yPrediction = linearRegressor.predict(xTest)

# Visualization
'''''''''
df = pandas.DataFrame({'Actual': yTest.flatten(), 'Predicted': yPrediction.flatten()})
print(df)

df1 = df
df1.plot(kind='bar')
plot.show()

'''''''''''

plot.scatter(xTest, yTest,  color='gray')
plot.plot(xTest, yPrediction, color='red', linewidth=2)
plot.show()

# Displaying errors
print('Mean Absolute Error:', metrics.mean_absolute_error(yTest, yPrediction))  
print('Mean Squared Error:', metrics.mean_squared_error(yTest, yPrediction))  
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(yTest, yPrediction)))
