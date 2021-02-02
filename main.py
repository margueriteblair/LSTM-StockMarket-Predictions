import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json
from sklearn.preprocessing import MinMaxScaler


#pandas.read_csv will return a dataframe object
df = pd.read_csv("Stocks/aapl.us.txt", delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
print("Loaded data from the Kaggle dataset")
rows, columns = df.shape
df = df.sort_values('Date')
print(df.isna().any()) #when we print this, we're looking for all false values, which means that all values match definitions
print(df.head()) #df.head() prints first 5 results, df.tail() will print last 5 results
# print(df.info()) all our datatypes are homogenous as needed
plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
# plt.show()
high_prices = df.loc[:, 'High'].to_numpy()
low_prices = df.loc[:, 'Low'].to_numpy()
mid_prices = (high_prices + low_prices) / 2.0

split = int(round(0.85*rows))
train_data = mid_prices[:split]
test_data = mid_prices[split:]

EMA = 0.0
gamma = 0.1
for ti in range(split):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

all_mid_data = np.concatenate([train_data, test_data], axis=0)

window_size = 100
N = all_mid_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
    date = df.loc[pred_idx, 'Date']
    std_avg_predictions.append(np.mean(all_mid_data[pred_idx - window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - all_mid_data[pred_idx]) ** 2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

#In other words, you say the prediction at t+1 is the average value of all the stock prices you observed within a window of t to t−N.
plt.figure(figsize=(18, 9))
# plt.plot(7125, next_day, 'ro')
plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
plt.title('SMA Prediction vs Actual')
# plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
# plt.show()


window_size = 100
N = all_mid_data.size
run_avg_predictions = []
run_avg_x = []
mse_errors = []
running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1, N):
    running_mean = running_mean * decay + (1.0 - decay) * all_mid_data[pred_idx - 1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1] - all_mid_data[pred_idx]) ** 2)

print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))
plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.title('EMA Prediction vs Actual')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.legend(fontsize=18)
plt.show()

#below is going to be a separate data cleaning and prediction model:
#we need to scale down numbers because the distance calculation that happens
#in the training phase needs to be mitigated to create more accurate results

#below is where we start the data preprocessing stage. data preprocessing involves
#data cleaning, transformation, integration etc. once the data is clean, we will divide into test & training sets
training_set = df['Open']
training_set = pd.DataFrame(training_set)

sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit(training_set)

#below, we're going to create a data structure w 60 timestamps
#and 1 output
X_train = []
y_train = []
for i in range(60, mid_prices.size):
    X_train.append(train_data_scaled[i-60:i, 0])
    y_train.append(train_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential #We can create a sequential model by passing a list through it
from keras.layers import Dense #Regularly connected NN layer, reps a matrix vector
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
#the above initializes the neural network

#now we train the neural network for prediction
#this step is the most important
#data is fed to the RNN and is trained for prediction assigning random biases and weights

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #regularization technique

#now we are adding a second LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#adding a third layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#adding a fourth LSTM layer and once again, some droout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#we only want/need one output so we set the output layer last and have units set to one
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
#the optimizer is one of the required values here for the compiler
#we have to make sure that the weights dont get too large
#we have to regularize the data
#if the data becomes to large, it's then overfit for the model
#dropouts are used in making the neurons morerobust and allowingthem to predict the trends without focusing on any one neuron
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
#an epoch in machine learning is a passage of time, when we see all the epochs print out
#it's simply a reflection of all the passes of the algorithm
#now we make predctions and visualize results
dataset_test = pd.read_csv("Stocks/aapl.us.txt", index_col="Date", parse_dates=True)
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_test["Volume"] = dataset_test["Volume"].str.repace(',', '').astype(float)

test_set = dataset_test["Open"]
test_set = pd.DataFrame(test_set)

test_set.info()
dataset_total = pd.concat((df['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) -60].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price)

plt.plot(real_stock_price, color="red", label="Actual")
plt.plot(predicted_stock_price, color="blue", label="Predicted")
plt.title("Stock Price Prediction Using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()




