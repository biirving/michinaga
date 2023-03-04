"""
MACD will be an indicator that I initially focus on

MACD + RSI

MACD + MFI

Woe to the conquered.

"""

import numpy as np
import torch
from torch import nn, tensor
import talib
import matplotlib.pyplot as plt
import math
import ta
from ta.trend import MACD 
from ta.momentum import rsi
import pandas as pd

#adjusted closing prices
googl_data = torch.load('sp500/GOOG.pt').numpy().astype('double')[:, 3]
macd, macd_signal, macd_hist = talib.MACD(np.flip(googl_data)[0:100], fastperiod = 12, slowperiod = 26, signalperiod = 9)  

#signal = talib.signal(googl_data)

def plot(x1, y1, x2, y2):
    plt.plot(x1, y1, label = 'MACD')
    plt.plot(x2, y2, label = 'Signal')
    plt.legend()
    plt.show()

print(macd[33:].shape)
nans = 0
for n in macd:
    if(math.isnan(n)):
        nans += 1
print('nans: ', nans)

#plot(np.arange(100), macd, np.arange(100), macd_signal)

#print(talib.get_functions())

# macd attempt 2

macd_2 = MACD(pd.DataFrame(np.flip(googl_data), columns=['close'])['close'], window_fast=12,window_slow =26,window_sign=9, fillna=False)
macd_obj = macd_2.macd()
macd_hist = macd_2.macd_diff()
macd_sig = macd_2.macd_signal()
index = 0 
nans =0
for n in macd_obj.values:
    if(math.isnan(n)):
        nans += 1
    index += 1
print('nans: ', nans)

print(macd_obj.values)
print(macd_sig.values)
nans = 0
for n in macd_sig.values:
    if(math.isnan(n)):
        nans += 1
print('nans: ', nans)

#plot(np.arange(len(macd_obj.values)), macd_obj.values, np.arange(len(macd_obj.values)), macd_sig.values[0:100])

# so then, how are we to proceed?
# let it rip
# so, what is our general strategy? USE ATTENTION AND DEEP LEARNING TO CREATE PROFIT

# rsi
rsi_1 = rsi(pd.DataFrame(np.flip(googl_data)[0:100], columns=['close'])['close'], window=14, fillna = False)
print(rsi_1[:].values)
"""

MACD BASIC STRATEGY
Buy: 𝑀𝑎𝑐𝑑𝑡−1 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 > 0)
Sell: 𝑀𝑎𝑐𝑑𝑡−1 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 < 0)

MACD --> signal crossover AND (∀{𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 } ≤ 𝐿𝑜𝑤𝑒𝑟 𝑇ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑
so, how would we calculate these statistics on a dataset? 

"""

# we want to feed the non-nan values into the model.

def extract_macd_rsi_data(macd, macd_signal, rsi):
    # macd signal crossover combined with the fact that the the 5 previous rsi's have fallen below the lower threshold
    # the lower threshold is 30
    lower_threshold = 30
    for x in range(rsi.shape[0]):
        toCheck = rsi[x:x+5]
        # then this rsi chuck is valid
        # are we going to save these as tensors?
        if(not max(toCheck) > 30):
            # a BUY signal
            if(macd[x - 1] < macd_signal[x - 1] and (macd[x] > macd_signal[x] and macd[x] > 0)):


    pass
