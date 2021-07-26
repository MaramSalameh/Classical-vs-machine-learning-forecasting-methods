# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:31:31 2020

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
plt.plot(residential_data['Residential Price'])
plt.title('Residential Price')
plt.xlabel('Year')
plt.ylabel('Price in in Cents/kWh')

############ residiential data prices prediction#####################################
################Select target and assign training and testing sets
res_data_prices = residential_data.drop(['Residential Revenue', 'Residential Customers', 'Residential Sales'], axis = 1, inplace = False)

res_data_prices.index.freq = 'MS'

prices_train = res_data_prices.iloc[:116, 0]
prices_test = res_data_prices.iloc[116:, 0]

### seasonal decompose
prices_decompose = seasonal_decompose(res_data_prices)
prices_decompose.plot()

#######################Implement models
###Holt-Winter's
############# model 3
prices_model3 = ExponentialSmoothing(prices_train, seasonal='add', trend='mul', seasonal_periods=12).fit()
prices_model3.params
prices_model3.summary()
pred_price3 = prices_model3.predict(start=prices_test.index[0], end=prices_test.index[-1])
#plt.plot(prices_train.index, prices_train, label='Train')
plt.plot(prices_test.index, prices_test, label='Test')
plt.plot(pred_price3.index, pred_price3, label='Predicitions')
plt.legend(loc=2)
plt.title('Holt-Winters Model for Electricity Prices')
plt.xlabel("year/Month")
plt.ylabel('Prices in Cents/kwh')

rmse_price3 = math.sqrt(mean_squared_error(prices_test, pred_price3))
rmse_price3 #1.1688817871587398
mae_price3 = mean_absolute_error(prices_test, pred_price3)
mae_price3  #0.9551255927083537
pred_price2021_3 = prices_model3.predict(start='2020-01-01',end='2020-12-01')

############ model 4
prices_model4 = ExponentialSmoothing(prices_train, seasonal='add', trend='add', seasonal_periods=12).fit()
prices_model4.params
prices_model4.summary()
pred_price4 = prices_model4.predict(start=prices_test.index[0], end=prices_test.index[-1])
#plt.plot(prices_train.index, prices_train, label='Train')
plt.plot(prices_test.index, prices_test, label='Test')
plt.plot(pred_price4.index, pred_price4, label='Predicitions')
plt.legend(loc=2)
plt.title('Holt Winters Model For Electricity Prices')
plt.xlabel("Year/Month")
plt.ylabel('Price in Cents/kwh')

rmse_price4 = math.sqrt(mean_squared_error(prices_test, pred_price4))
rmse_price4 #1.152848195494752
mae_price4 = mean_absolute_error(prices_test, pred_price4)
mae_price4  #0.9195042092070094

### make future predicitions
pred_price2020_4 = prices_model4.predict(start='2020-08-01',end='2020-12-01')

#**** Model 4 has best accuracy

####################################################################################
#### 2. ARIMA
dftest = adfuller(res_data_prices)
dftest
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
#p value = 0.994777 > 0.05, therefore its not stationary

#Estimating & Eliminating Trend and Seasonality - use log() transformation
#res_prices_log = np.log(res_data_prices)
#res_prices_log
#plt.plot(res_prices_log)

#decomposition_resprice1 = seasonal_decompose(res_prices_log)
#decomposition_resprice1.plot()
#### make data stationary

#res_prices_log
res_prices_diff = res_data_prices - res_data_prices.shift()
res_prices_diff
plt.plot(res_prices_diff)
res_prices_diff.dropna(inplace=True)

price_af = adfuller(res_prices_diff)
dfoutput = pd.Series(price_af[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
#p-value = 7.256920e-18

## seasonal diff
res_s_diff = res_prices_diff - res_prices_diff.shift(12)
res_s_diff
plt.plot(res_s_diff)
res_s_diff.dropna(inplace = True)

s_af = adfuller(res_s_diff)
dfoutput = pd.Series(s_af[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
### p-value  0.000106

decom = seasonal_decompose(res_s_diff)
decom.plot()

plot_acf(res_s_diff)
plot_pacf(res_s_diff)


stepwise_fit = auto_arima(res_data_prices, start_p = 1, start_q = 1, 
                          max_p = 10, max_q = 10, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    
                          suppress_warnings = True,  
                          stepwise = True)            
# To print the summary 
stepwise_fit.summary()
stepwise_fit.plot_diagnostics()

### model arima
res_prices_model = SARIMAX(prices_train, order=(0,0,0), seasonal_order=(1,1,0,12)).fit(disp=-1)

ARIMA_pred_res_prices = res_prices_model.predict(start=prices_test.index[0],end=prices_test.index[-1])
ARIMA_pred_res_prices

#plt.plot(prices_train.index, prices_train, label='Train')
plt.plot(prices_test.index, prices_test, label='Test')
plt.plot(ARIMA_pred_res_prices.index, ARIMA_pred_res_prices, label='Predictions')
plt.title('SARIMA Model for Electricity Prices')
plt.legend(loc=2)
plt.xlabel("Year/Month")
plt.ylabel('Price in Cents/kwh')

####check error scores
res_rmse_pricesAR1 = math.sqrt(mean_squared_error(prices_test, ARIMA_pred_res_prices))
res_rmse_pricesAR1 #1.268979789600164
res_mae_pricesAR1 = mean_absolute_error(prices_test, ARIMA_pred_res_prices)
res_mae_pricesAR1  #1.0560270090103516

### predict for 2020-2021
pred_res_prices2020 = res_prices_model.predict(start='2020-08-01',end='2020-12-01')
pred_res_prices2020

########################################################################################
#### 3. Linear Regression
X_prices = residential_data.iloc[:,[0,1,2]].values
Y_prices = residential_data.iloc[:,3].values

X_train_prices = X_prices[:116, :]
X_test_prices = X_prices[116:, :]
y_train_prices = Y_prices[:116]
y_test_prices = Y_prices[116:]

price_reg = linear_model.LinearRegression().fit(X_train_prices, y_train_prices)

price_y_predictions = price_reg.predict(X_test_prices)
print(price_y_predictions)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_prices, price_y_predictions))
#Mean Absolute Error:  0.49934266635342367
print('Mean Squared Error:', metrics.mean_squared_error(y_test_prices, price_y_predictions))
#Mean Squared Error: 0.43221620149028706
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_prices, price_y_predictions)))
#Root Mean Squared Error: 0.6574315184795197

print(price_reg.coef_)
#[ 1.35713237e-05 -2.38566585e-06  3.65098538e-08]

print(price_reg.intercept_)
#[17.003786976779438]

print(price_reg.score(X_test_prices,y_test_prices))
#0.8379867674822591

### set index and plot
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(y_test_prices, index)
ts_pred_test = pd.DataFrame(price_y_predictions, index)


plt.plot(ts_test.index, y_test_prices, label= 'Test')
plt.plot(ts_pred_test.index,price_y_predictions, label='Predictions')
plt.title('Linear Regression Model for Electricty Prices')
plt.legend(loc=2)
plt.xlabel("Year/Month")
plt.ylabel('Price in Cents/kwh')

################ Random Forest
X_prices = residential_data.iloc[:,[0,1,2]].values
Y_prices = residential_data.iloc[:,3].values

X_train_prices = X_prices[:116, :]
X_test_prices = X_prices[116:, :]
y_train_prices = Y_prices[:116]
y_test_prices = Y_prices[116:]



####Scale data
sc = StandardScaler()
X_train_prices = sc.fit_transform(X_train_prices)
X_test_prices = sc.transform(X_test_prices)

## fit and train model
regressor_price = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_price.fit(X_train_prices, y_train_prices)
y_pred_price = regressor_price.predict(X_test_prices)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_prices, y_pred_price))
#Mean Absolute Error: 1.255291666666672

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_prices, y_pred_price)))
#Root Mean Squared Error:  1.4838314496038105


### set index and plot data
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(y_test_prices, index)
ts_pred_test = pd.DataFrame(y_pred_price, index)


plt.plot(ts_test.index, y_test_prices, label= 'Test')
plt.plot(ts_pred_test.index,y_pred_price, label='Predictions')
plt.title('Random Forest Model for Electricty Prices')
plt.legend(loc=2)
plt.xlabel("Year/Month")
plt.ylabel('Price in Cents/kwh')
#######################################################3
#############LSTM
X = res_data_prices.values

#Divide your data into input (X) and output (y) components with time steps=1
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

supervised = timeseries_to_supervised(X, 1)
supervised.head()
res_data_prices.head()

### #Transform Time Series to Stationary
#create a differenced series and then divide data into input (X) and output (y) components with time steps=1
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform to be stationary
diff_values = difference(X, 1)
diff_values.head()

#invert this process in order to take forecasts made on the differenced series back into their original scale
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Divide the transformed data into input (X) and output (y) components with time steps=1
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
supervised_values

train, test = supervised_values[0:-12], supervised_values[-12:]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)
scaled_series = Series(scaled_X[:, 0])
scaled_series.head()

inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
inverted_series.head()

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

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

neurons=4
batch_size=1
nb_epoch=3000

layer = LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True)

model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
#model.add(Activation('softmax'))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(nb_epoch):
	model.fit(X, y, epochs=2, batch_size=batch_size, shuffle=False)
	model.reset_states()
    
# make a one-step forecast
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

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(res_data_prices.values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = res_data_prices.values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(res_data_prices.values[-12:], predictions))
print('Test RMSE: %.3f' % rmse) #Test RMSE: 1.48

mae = mean_absolute_error(res_data_prices.values[-12:], predictions)
mae #1.25

# line plot of observed vs predicted
index = pd.date_range(start='2019-08-01', periods=12, freq='M')
ts_test = pd.DataFrame(res_data_prices.values[-12:], index)
ts_pred_test = pd.DataFrame(predictions, index)


plt.plot(ts_test.index, res_data_prices.values[-12:], label= 'Test')
plt.plot(ts_pred_test.index,predictions, label='Predictions')
plt.title('LSTM Model for Electricty Prices')
plt.legend(loc=2)
plt.xlabel("Year/Month")
plt.ylabel('Price in Cents/kwh')
