import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plot
# from pandas_datareader import data
import datetime as dt
import urllib.request, json
import tensorflow as tf
from sklearn.preprocessing import MaxMinScaler

#stocks come in different flavors!
#open
#close
#high
#low

data_source = 'kaggle'

if data_source == 'alphavantage':
    api_key="KQUV4B3CT1Z898PZ"
    ticker = "AAL"
    url_string="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save="stock_market_data-%s.csv"%ticker
