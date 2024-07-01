pip install pgmpy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
From google.colab import files
uploaded = files.upload()

import io
dataset_train = pd.read_csv(io.BytesIO(uploaded['Google_Stock_Price_Train.csv']))
dataset_train.head()
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
print(training_set)
print(training_set.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

xtrain = []
ytrain = []
for i inrange(60, 1258):
xtrain.append(scaled_training_set[i-60:i,0])

ytrain.append(scaled_training_set[i,0])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
xtrain.shape

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(xtrain, ytrain, epochs=100, batch_size = 32)
uploaded = files.upload()

dataset_test = pd.read_csv(io.BytesIO(uploaded['Google_Stock_Price_Test.csv']))
actual_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
xtest = []
foriin range (60,80):
xtest.append(inputs[i-60:i, 0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

predicted_stock_price = regressor.predict(xtest)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label="Predicted Google Stock Price")
plt.title('Google Stock Price Prediction')

plt.xlabel("Time")
plt.ylabel('Google Stock Price')
plt.legend()
