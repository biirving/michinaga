import requests
import csv
import pandas as pd
import numpy as np
import torch

# the nasdaq tickers (as of Feb 1, 2023)
names = np.load('nasdaq_tickers.npy')
print(names)
key = 'ONRD3KINP1JRCPSC'

new_price_data = {}

for name in names:
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + name + '&outputsize=compact&apikey=' + key
    r = requests.get(url)
    data = r.json()
    print(name)
    print(data)
    print(name, data['Time Series (Daily)'])

    price_data = None
    count = 0
    for day in data['Time Series (Daily)']:
        today = data['Time Series (Daily)'][day]
        onThisDay = torch.tensor([float(today['1. open']), float(today['2. high']), float(today['3. low']),
        float(today['5. adjusted close']), float(today['6. volume'])])
        if(price_data is None):
            price_data = onThisDay.view(1, 5)
        else:
            price_data = torch.concat((price_data, onThisDay.view(1, 5)), 0)

    new_price_data[name] = price_data



