import pandas as pd
import numpy as np
import os
#os is a standard import in python that provides us with functions for interacting with the operating system
#for instance, os.name will return the name of the os that one is using
import matplotlib.pyplot as plt
#mathplotlib is a comprehensive library used for making data vizualizations in python
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

#because our application runs on launch, we'll just toggle the datasource
#by literally editing it below
data_source = 'kaggle'

if data_source == 'alphavantage':
    #nb: alphasource isn't fleshed out yet!
    #working on the kaggle dataset first
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
    #i'm going to break it down into the nitty gritty
    #we're saying that our dataframe is equal to the pandas method read_csv
    #within read_csv() we're passing a filepath using 'os' module
    #although the delimiter by default is ',' we'll explicitly declare our delimiter
    #we'll also pass in a parameter for usecols, which accepts a list-like arg
    #use cols can either be integer based or column names
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print("Loaded data from the Kaggle repository")

    #pandas.DataFrame.sort_values() accepts a string, which is a name or list of names to sort by
    #we want to sort the dataframe by date
    df = df.sort_values('Date')
    #double check the result
    # print(df.head())
    #pandas.dataFrame.head() returns the first n rows of a dataframe
    #the default is 5 when left blank
    #matplotlib.pyplot.figure() makes a new figure!
    #figsize is the width and height in inches
    plt.figure(figsize=(18, 9))
    #pyplot.plot() plots y vs x as either lines or markers
    #we're saying we want to plot as many x values are there are rows in the first column
    #and for the y values, we're plotting the mid price for all entries
    #that's why we're diving the high + low / 2
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    #the two following lines allow us to label the x and y axis and provide a font size
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    # plt.show()
    #commented out the above to see other mathplotlib
    #pandas.dataFrame.to_numpy() will convert a dataframe to a numpy array
    #we want to do this for both the high and the low prices so that we're effectively able to get a mid price
    #pandas.dataFrame.loc() is used to access a group of rows or columns by labels or boolean array
    #pandas.dataFrame.iloc() does a similar function but instead with integer values
    #I can't quite figure out why, but I need that colon separating the high and low
    high_prices = df.loc[:, 'High'].to_numpy()
    low_prices = df.loc[:, 'Low'].to_numpy()
    mid_prices = (high_prices + low_prices) / 2.0

    #we're splitting the data directly in half to create a training and a test set
    #although the standard is more typically 80:20
    test_data = mid_prices[11000:]
    train_data = mid_prices[:11000]

    #we'll be using MinMaxScaler to scale all the data to be in the region between 0 & 1
    #we also reshape the test and train data to be in the shape
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)
    #When you scale data, you need to scale both the test data and the training data
    #We're going to normalize data by breaking the full series of data into windows
    smoothing_window_size = 2500
    #we're saying for di in range 0-10000 and we're incrementing by 2500 upwards
    #it then reformats each increments of that data
    for di in range(0, 10000, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di + smoothing_window_size:, :])
    train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

    #now we reshape the train and test data back to the shape of [data_size]
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

    #one step ahead prediction via averaging:
    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    #mse stands for mean squared errors
    mse_errors = []
    #wedo this before moving on to long short term mem models

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
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    # plt.show()

    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    # plt.legend(fontsize=18)
    plt.show()

    #all of the above is before we incorporate LST modeling
    #Long Short-Term Memory (LSTM) models are extremely powerful time-series models.
    # A LSTM can predict an arbitrary number of steps into the future.
    # A LSTM module (or a cell) has 5 essential components which allows them to model both long-term and short-term data.
    #The above code only expects one step into the future, but with LSTM we can predict multiple steps into the future

