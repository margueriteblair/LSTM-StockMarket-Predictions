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