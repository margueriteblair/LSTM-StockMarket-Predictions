import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pandas_datareader import data
import datetime as dt
import urllib.request, json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
0

df = pd.read_csv("Stocks/aapl.us.txt", delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
print("Loaded data from the Kaggle dataset")
rows, columns = df.shape
df = df.sort_values('Date')
df.isna().any()
# print(df.head())

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
# print(7125-N)
# t_plus_one = 7125
# diff = t_plus_one - (t_plus_one-N)
# next_day = np.mean(all_mid_data[diff: t_plus_one])
# print(next_day)
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


