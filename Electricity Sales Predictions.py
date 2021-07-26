# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:31:10 2020

@author: MS
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima 
from statistics import mean
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import theano
import tensorflow
import keras
import scipy
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt



#read data file
elec_data = pd.read_excel('sales_revenue.xlsx', skiprows=[0,2,18771])

#assign columns names
elec_data.columns = ['Year', 'Month', 'State','Data Status', 'Residential Revenue', 'Residential Sales', 'Residential Customers', 'Residential Price', 
             'Commercial Revenue', 'Commercial Sales', 'Commercial Customers', 'Commercial Price',
             'Industrial Revenue', 'Industrial Sales', 'Industrial Customers', 'Industrial Price',
             'Transportation Revenue', 'Transportation Sales', 'Transportation Customers', 'Transportation Price',
             'Other Revenue', 'Other Sales', 'Other Customers', 'Other Price',
             'Total Revenue', 'Total Sales', 'Total Customers', 'Total Price']


# drop irrelevant columns
elec_data.drop(['Data Status','Other Revenue', 'Other Sales', 'Other Customers', 'Other Price',
             'Total Revenue', 'Total Sales', 'Total Customers', 'Total Price'], axis=1, inplace=True)

#select data from dataframe
CA_datasubset = elec_data[(elec_data.Year >= 2010) & (elec_data.State == 'CA')]

#check data types
CA_datasubset.dtypes

#Year                          int64
#Month                         int64
#State                        object
#Residential Revenue         float64
#Residential Sales           float64
#Residential Customers        object
#Residential Price           float64
#Commercial Revenue           object
#Commercial Sales             object
#Commercial Customers         object
#Commercial Price             object
#Industrial Revenue           object
#Industrial Sales             object
#Industrial Customers         object
#Industrial Price             object
#Transportation Revenue       object
#Transportation Sales         object
#Transportation Customers     object
#Transportation Price         object
 
#Change data types
datatypes_dict = {'Residential Customers' : 'int64', 'Commercial Revenue': float, 'Commercial Sales': float, 
                  'Commercial Customers': 'int64', 'Commercial Price': float, 'Industrial Revenue': float, 
                 'Industrial Sales': float, 'Industrial Customers': 'int64', 'Industrial Price': float, 
                 'Transportation Revenue' : float, 'Transportation Sales': float, 'Transportation Customers' : 'int64',
                'Transportation Price': float}

CA_datasubset = CA_datasubset.astype(datatypes_dict)
CA_datasubset.dtypes

#Year                          int64
#Month                         int64
#State                        object
#Residential Revenue         float64
#Residential Sales           float64
#Residential Customers         int64
#Residential Price           float64
#Commercial Revenue          float64
#Commercial Sales            float64
#Commercial Customers          int64
#Commercial Price            float64
#Industrial Revenue          float64
#Industrial Sales            float64
#Industrial Customers          int64
#Industrial Price            float64
#Transportation Revenue      float64
#Transportation Sales        float64
#Transportation Customers      int64
#Transportation Price        float64

#change data into time series data 
Date = pd.to_datetime(CA_datasubset[['Year', 'Month']].assign(DAY=1))
CA_datasubset.insert(0, 'Date', Date)

#set index
CA_datasubset.set_index('Date', inplace = True)
CA_datasubset.index

#remove irrelevant columns
CA_datasubset.drop(['Year', 'Month', 'State'], axis = 1, inplace = True)

#Exploratory Analysis
CA_datasubset.info()
CA_datasubset.shape #(128, 16)
CA_datasubset.describe()
CA_datasubset.head()
CA_datasubset.tail()

# Revenue is Thousand Dollars
# Sales is in Megawatthours
# Price in in Cents/kWh

##################### CA Residential data visualization ##########
#Select residential data from subset
residential_data = CA_datasubset.iloc[:, 0:4]

#Plot historgram and scatter matrix of resdiential data
residential_data.hist()
scatter_matrix(residential_data, figsize=(15, 15))

#Plot of Residential Sales
plt.plot(residential_data['Residential Sales'])
plt.title('Residential Consumption/Sales')
plt.xlabel('Year')
plt.ylabel('Consumption in Megawatthours')


####################### residential data sales/consumption prediction #################
#######################################################################################################

##### select target variable and assign training and testing sets
res_data_sales = residential_data.drop(['Residential Revenue', 'Residential Customers', 'Residential Price'], axis = 1, inplace = False)
res_data_sales.index.freq = 'MS'

### select 116 data points for training 
sales_train = res_data_sales.iloc[:116, 0]
sales_test = res_data_sales.iloc[116:, 0]

res_data_sales.max()
res_data_sales.min()

#######Implemend forecasting models###############

###1. Holt winters

sales_decompose = seasonal_decompose(res_data_sales)
sales_decompose.plot()


##########model 1

res_model1 = ExponentialSmoothing(sales_train, seasonal='add', trend=None, seasonal_periods=12).fit()
res_model1.params
res_model1.summary()
pred1 = res_model1.predict(start=sales_test.index[0], end=sales_test.index[-1])
#plt.plot(sales_train.index, sales_train, label='Train')
plt.plot(sales_test.index, sales_test, label='Test')
plt.plot(pred1.index, pred1, label='Predicitions')
plt.legend(loc=2)
plt.title('Holt-winters Model for Electricity Sales')
plt.xlabel("year")
plt.ylabel('Sales in Megawatthours')

rmse1 = math.sqrt(mean_squared_error(sales_test, pred1))
rmse1 #703559.4706101603
mae1 = mean_absolute_error(sales_test, pred1)
mae1  #589974.6780930416
pred2020 = res_model1.predict(start='2020-09-01',end='2020-12-01')


#####model 2

###train and fit model
res_model2 = ExponentialSmoothing(sales_train, seasonal='mul', trend=None, seasonal_periods=12).fit()
res_model2.params
res_model2.summary()
pred2 = res_model2.predict(start=sales_test.index[0], end=sales_test.index[-1])
#plt.plot(sales_train.index, sales_train, label='Train')
plt.plot(sales_test.index, sales_test, label='Test')
plt.plot(pred2.index, pred2, label='Predicitions')
plt.legend(loc=2)
plt.title('Holt-winters Model for Electrcity Sales')
plt.xlabel("year/Month")
plt.ylabel('Consumption in Megawatthours')

### Check error scores
rmse2 = math.sqrt(mean_squared_error(sales_test, pred2))
rmse2 #702951.1261497248
mae2 = mean_absolute_error(sales_test, pred2)
mae2 #589194.503721711

## future predictions
pred2020_2 = res_model2.predict(start='2020-09-01',end='2021-12-01')

#*** multipicative Model has better accuracy rate

################################################################################################
######## 2. Seasonal ARIMA

#Plot rolling statistics:
rolmean =  pd.Series(sales_train).rolling(window=12).mean() 
rolmean
rolstd = pd.Series(sales_train).rolling(window=12).std()
rolstd
orig = plt.plot(sales_train, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc=2)
plt.title('Rolling Mean & Standard Deviation')

###check if stationary
dftest = adfuller(sales_train)
dftest
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
#p value = 0.056316 > 0.05 

###differencing
ressale_s_diff = res_data_sales - res_data_sales.shift(1)
ressale_s_diff 
plt.plot(ressale_s_diff)
ressale_s_diff.dropna(inplace = True)

##adfuller test; check if stationary after differencing
dftest = adfuller(ressale_s_diff)
dftest
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
#p-value =

#### plot ACF and PACF
plot_acf(ressale_s_diff, lags= 40)
plot_pacf(ressale_s_diff, lags = 40)

### find best ARIMA model
stepwise_fit = auto_arima(sales_train, start_p = 1, start_q = 1, 
                          max_p = 5, max_q = 5, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    
                          suppress_warnings = True,  
                          stepwise = True)            
# To print the summary 
stepwise_fit.summary()
print(stepwise_fit.aic())
stepwise_fit.plot_diagnostics()

## model sarimax
res_sales_model1 = SARIMAX(sales_train, order=(0,0,0), seasonal_order=(0,1,0,12)).fit(disp=-1)
res_sales_model1.params
res_sales_model1.summary()

ARIMA_pred_res_sales = res_sales_model1.predict(start=sales_test.index[0],end=sales_test.index[-1])
ARIMA_pred_res_sales

#plt.plot(sales_train.index, sales_train, label='Train')
plt.plot(sales_test.index, sales_test, label='Test')
plt.plot(ARIMA_pred_res_sales.index, ARIMA_pred_res_sales, label='Predicitions')
plt.title('SARIMA Model for Residential Sales')
plt.legend(loc=2)
plt.xlabel("year/Month")
plt.ylabel('Consumption in Megawatthours')

### check error rates
res_rmse_salesAR1 = math.sqrt(mean_squared_error(sales_test, ARIMA_pred_res_sales))
res_rmse_salesAR1 #876140.9375907762
res_mae_pricesAR1 = mean_absolute_error(sales_test, ARIMA_pred_res_sales)
res_mae_pricesAR1  #716207.1083333334

### predict for 2020-2021
pred_res_sales2020 = res_sales_model1.predict(start='2020-09-01',end='2020-12-01')
pred_res_sales2020

################################################################################################
###########3. Linear Regression
X = residential_data.iloc[:,[0,2,3]].values
Y = residential_data.iloc[:,1].values

X_train = X[:116, :]
X_test = X[116:, :]
y_train = Y[:116]
y_test = Y[116:]

## train anf fit model
reg = linear_model.LinearRegression().fit(X_train, y_train)
#reg.fit(train_X, train_Y)

## make predicitons
y_predictions = reg.predict(X_test)
print(y_predictions)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predictions))
lg_mae = mean_absolute_error(y_test, y_predictions)
lg_mae
#Mean Absolute Error: 186132.59209643863
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predictions))
#Mean Squared Error: 73206840090.98055
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predictions)))
lg_rmse = math.sqrt(mean_squared_error(y_test, y_predictions))
lg_rmse #240209.89838173168
#Root Mean Squared Error: 240209.89838173168

print(reg.coef_)
#[ 5.58987966e+00 -1.45077666e-02 -3.94263828e+05]

print(reg.intercept_)
#[7232206.025894092]

print(reg.score(X_test,y_test))
#0.9775328838865265

### set index and plot predicitons
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(y_test, index)
ts_pred_test = pd.DataFrame(y_predictions, index)


plt.plot(ts_test.index, y_test, label= 'Test')
plt.plot(ts_pred_test.index,y_predictions, label='Predicitions')
plt.title('Linear Regression Model for Electricty Sales')
plt.legend(loc=2)
plt.xlabel("year/Month")
plt.ylabel('Consumption in Megawatthours')


#################################################################################################
#### 4. random forest sales

X = residential_data.iloc[:,[0,2,3]].values
Y = residential_data.iloc[:,1].values

X_train = X[:116, :]
X_test = X[116:, :]
y_train = Y[:116]
y_test = Y[116:]


#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

### check error scores
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Root Mean Squared Error: 608674.1444280639
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

#Mean Absolute Error: 484644.95329166896



### set index and plot data
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(y_test, index)
ts_pred_test = pd.DataFrame(y_pred, index)


plt.plot(ts_test.index, y_test, label= 'Test')
plt.plot(ts_pred_test.index,y_pred, label='Predicitions')
plt.title('Random Forest Model for Electricty Sales')
plt.legend(loc=2)
plt.xlabel("year/Month")
plt.ylabel('Consumption in Megawatthours')

#######################################################################
##############################################################################
#CA_datasubset
### load data as array
#values = residential_data.values

## normalize features
#scaler = MinMaxScaler()
#scaled = scaler.fit_transform(values)

### transform to supernised learning
#reframed = timeseries_to_supervised(scaled,1)


#X = scaled[:,[0,2,3]]
#Y = scaled[:,1]

#X_train = X[:116, :]
#X_test = X[116:, :]
#y_train = Y[:116]
#y_test = Y[116:]



########################### 5. LSTM
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

X_sales = res_data_sales.values

supervised_sales = timeseries_to_supervised(X_sales, 1)
supervised_sales.head()
res_data_sales.head()

diff_values_sales =  difference(X_sales, 1) ### try 1
diff_values_sales.head()

# Divide the transformed data into input (X) and output (y) components with time steps=1
supervised_sales = timeseries_to_supervised(diff_values_sales, 1)
supervised_values_sales = supervised_sales.values
supervised_values_sales

# split data into train and test sets, use 20 of them as testing set and the rest as training 
train_sales, test_sales = supervised_values_sales[0:-12], supervised_values_sales[-12:]

scaler_sales = MinMaxScaler(feature_range=(-1, 1))
scaler_sales = scaler_sales.fit(X_sales)
scaled_X_sales = scaler_sales.transform(X_sales)
scaled_series_sales = Series(scaled_X_sales[:, 0])
scaled_series_sales.head()

inverted_X_sales = scaler_sales.inverse_transform(scaled_X_sales)
inverted_series_sales = Series(inverted_X_sales[:, 0])
inverted_series_sales.head()

# transform the scale of the data
scaler_sales, train_scaled_sales, test_scaled_sales = scale(train_sales, test_sales)

X_sales, y_sales = train_scaled_sales[:, 0:-1], train_scaled_sales[:, -1]
X_sales = X_sales.reshape(X_sales.shape[0], 1, X_sales.shape[1])

neurons=4
batch_size=1
nb_epoch=3000

layer = LSTM(neurons, batch_input_shape=(batch_size, X_sales.shape[1], X_sales.shape[2]), stateful=True)

model_sales = Sequential()
model_sales.add(LSTM(neurons, batch_input_shape=(batch_size, X_sales.shape[1], X_sales.shape[2]), stateful=True))
model_sales.add(Dense(1))
model_sales.compile(loss='mean_squared_error', optimizer='adam')

for i in range(nb_epoch):
	model_sales.fit(X_sales, y_sales, epochs=1, batch_size=batch_size, shuffle=False)
	model_sales.reset_states()
    
predictions_sales = list()
for i in range(len(test_scaled_sales)):
	# make one-step forecast
	X_sales, y_sales = test_scaled_sales[i, 0:-1], test_scaled_sales[i, -1]
	yhat = forecast_lstm(model_sales, 1, X_sales)
	# invert scaling
	yhat = invert_scale(scaler_sales, X_sales, yhat)
	# invert differencing
	yhat = inverse_difference(res_data_sales.values, yhat, len(test_scaled_sales)+1-i)
	# store forecast
	predictions_sales.append(yhat)
	expected = res_data_sales.values[len(train_sales) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    
rmse_sales = sqrt(mean_squared_error(res_data_sales.values[-12:], predictions_sales))
print('Test RMSE: %.3f' % rmse_sales) #2007388.811

mae_sales = mean_absolute_error(res_data_sales.values[-12:], predictions_sales)
mae_sales #1583042.7093212113


### plot predictions vs test
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(res_data_sales.values[-12:], index)
ts_pred_test = pd.DataFrame(predictions_sales, index)


plt.plot(ts_test.index, res_data_sales.values[-12:], label= 'Test')
plt.plot(ts_pred_test.index,predictions_sales, label='Predictions')
plt.title('LSTM Model for Electricty Sales')
plt.legend(loc=2)
plt.xlabel("Year/Month")
plt.ylabel('Consumption in Megawatthours')

