import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 

## read an online csv file and display the first 5 rows 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'

df = pd.read_csv(path)
df.head()


## SOME BRIEF ANALYSIS AND VISUALIZATION TO HAVE AN OVERVIEW OF THE DATASET 
# summarize the data 
df.describe()


## extract the interested features (column names)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


## plot each of the features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


## plot against the CO2 emissions 
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

## plot cylinder vs emissions 
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()


#### CREATE TRAIN AND TEST DATASETS 

## Let's split our dataset into train and test sets. 
# 80% of the entire dataset will be used for training and 20% for testing.
# We create a mask to select random rows using np.random.rand() function


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# modeling

from sklearn import linear_model

regr = linear_model.LinearRegression()

## fit for the train 
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)


# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


## plot of the engine size vs co2 emissions with regression line 

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


## MODEL EVALUATION R2 SCORE

from sklearn.metrics import r2_score

# predict for the test 
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

# metrics
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y))) 
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

## Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature.

# Start by selecting FUELCONSUMPTION_COMB as the train_x data from the train dataframe, then select FUELCONSUMPTION_COMB as the test_x data from the test dataframe

regr_fuel = linear_model.LinearRegression()

train_x = train[['FUELCONSUMPTION_COMB']]
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr_fuel.fit(train_x, train_y)
#regr_fuel.fit(train_x,train_y)

#print('The coeff is: ', regr_fuel.coef_)
#print('the intercept is: ', regr_fuel.intercept_)

test_x = test[['FUELCONSUMPTION_COMB']]

test_y = test[['CO2EMISSIONS']]

predictions = regr_fuel.predict(test_x)

np.mean(np.absolute(predictions - test_y))






