import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# from pandas_datareader import data
import datetime as dt
import urllib.request, json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

#stocks come in different flavors!
#open
#close
#high
#low

data_source = 'kaggle'

if data_source == 'alphavantage':
    api_key = "KQUV4B3CT1Z898PZ"
    ticker = "AAL"
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = "stock_market_data-%s.csv"%ticker
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.item():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
            print("Data saved to : %s"%file_to_save)
            df.to_csv(file_to_save)
    else:
        print("File already exists. Leading data from CSV.")
        df = pd.read_csv(file_to_save)
else:
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print("Loaded data from the Kaggle repository")

    #we sort the dataframe by date
    df = df.sort_values('Date')
    #doublecheck the result
    df.head()
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()

    high_prices = df.loc[:, 'High'].as_matrix()
    low_prices = df.loc[:, 'Low'].as_matrix()
    mid_prices = (high_prices + low_prices) / 2.0

    test_data = mid_prices[11000:]
    train_data = mid_prices[:11000]

    #we'll be using MinMaxScaler to scale all the data to be in the region between 0 & 1
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)
    #When you scale data, you need to scale both the test data and the training data
    #We're going to normalize data by breaking the full series of data into windows
    smoothing_window_size = 2500
    for di in range(0, 1000, smoothing_window_size):
        scaler.fit(train_data[di: di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[train_data[di:di+smoothing_window_size:,:]])

    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    #now we rehshape the train and test data back to the shape of [data_size]
    train_data = train_data.reshape(-1)
    #normalize the test data
    test_data = scaler.transform(test_data).reshape(-1)

    #now perform the exponential moving average soothing
    #so the data will have a smoother curve than the original ragged data!
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    #the below is used for vizualization
    all_mid_data = np.concatenate([train_data, test_data], axis=0)

    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size, N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx, 'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()