import requests
import csv
import pandas as pd
import numpy as np

# the nasdaq tickers (as of Feb 1, 2023)
names = np.load('nasdaq_tickers.npy')

"""
Alpha-Vantage Data Extraction

Experimenting with the AlphaVantage API.
"""

key = 'ONRD3KINP1JRCPSC'

"""
Intraday Data
"""
symbol = 'AAPL'
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + symbol + '&interval=60min&outputsize=compact&apikey=' + key
r = requests.get(url)
data = r.json()
nas = data['Time Series (60min)']
print(len(nas))

"""
Trailing Data


CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=' + key


with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)

"""