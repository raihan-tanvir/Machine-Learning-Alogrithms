import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

dataset = pandas.read_csv('salaryData.csv')

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
lrPredict = linearRegressor.predict(xTest)

plt.scatter(xTest, yTest,  color='gray')
plt.plot(xTest, lrPredict, color='red', linewidth=3)
plt.show()

accuracy = linearRegressor.score(xTest, yTest)
print("Accuracy: {}%".format(int(round(accuracy * 100))))
lrAcc=int(round(accuracy * 100))

kf = KFold(n_splits=5)

print("\nLinear Regression:\n")
for train_index, test_index in kf.split(x):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    linearRegressor.fit(x_train, y_train)
    prediction = linearRegressor.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))  
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, prediction)))
    print('\n')
    #plt.scatter(x_test, y_test,  color='gray')
    #plt.plot(x_test, prediction, color='red', linewidth=2)
    #plot.show()



randForest = RandomForestRegressor(n_estimators=10,random_state=0)
randForest.fit(xTrain, yTrain)
rfPredict = randForest.predict(xTest)

plt.scatter(xTest, yTest,  color='gray')
plt.plot(xTest, rfPredict, color='blue', linewidth=2)
plt.show()

accuracy = randForest.score(xTest, yTest)
print("Accuracy: {}%".format(int(round(accuracy * 100))))
rfAcc=int(round(accuracy * 100))

print("Random Forest Regression:\n")
for train_index, test_index in kf.split(x):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    randForest.fit(x_train, y_train)
    prediction = randForest.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))  
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, prediction)))
    print('\n')
    #plt.scatter(x_test, y_test,  color='gray')
    #plt.plt(x_test, prediction, color='red', linewidth=2)
    #plt.show()
    
left = [1, 2] 
height = [lrAcc, rfAcc] 
tick_label = ['Linear', 'Random-Forest'] 
plt.bar(left, height, tick_label = tick_label, width = 0.5,color = ['blue', 'red']) 
  
plt.ylabel('Accuracy') 
plt.title('Linear vs Random-Forest') 

plt.show() 