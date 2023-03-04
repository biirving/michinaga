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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

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

class macdrsi:

    def __init__(self, tickers):
        self.tickers = tickers

    def plot(self, x1, y1, x2, y2):
        plt.plot(x1, y1, label = 'MACD')
        plt.plot(x2, y2, label = 'Signal')
        plt.legend()
        plt.show()

    def extract_macd(self, data, fast=12, slow=26, signal=9):
        macd_2 = MACD(pd.DataFrame(np.flip(data), columns=['close'])['close'], window_fast=fast, window_slow =slow, window_sign=signal, fillna=False)
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
        return macd_obj.values, macd_sig.values, macd_hist.values

    #plot(np.arange(len(macd_obj.values)), macd_obj.values, np.arange(len(macd_obj.values)), macd_sig.values[0:100])

    def get_rsi(self, data):
        # rsi
        rsi_1 = rsi(pd.DataFrame(np.flip(data)[0:100], columns=['close'])['close'], window=14, fillna = False)
        return rsi_1[:].values

    """

    MACD BASIC STRATEGY
    Buy: 𝑀𝑎𝑐𝑑𝑡−1 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 > 0)
    Sell: 𝑀𝑎𝑐𝑑𝑡−1 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 < 0)

    MACD --> signal crossover AND (∀{𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 } ≤ 𝐿𝑜𝑤𝑒𝑟 𝑇ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑
    so, how would we calculate these statistics on a dataset? 

    """

    def extract_macd_rsi_data(macd, macd_signal, rsi):
        labels = []
        # macd signal crossover combined with the fact that the the 5 previous rsi's have fallen below the lower threshold
        # the lower threshold is 30
        lower_threshold = 30
        for x in range(1, rsi.shape[0]):
            toCheck = rsi[x:x+5]
            if(not max(toCheck) > 30):
                # a BUY signal
                if(macd[x - 1] < macd_signal[x - 1] and (macd[x] > macd_signal[x] and macd[x] > 0)):
                    # we will just demarcate with binary classifications?
                    # the formation of the input tensors:
                    # 𝑀𝑎𝑐𝑑𝑡−1, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1, 𝑀𝑎𝑐𝑑𝑡, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡, 𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 
                    day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x], toCheck])
                    labels.append(torch.tensor([1, 0]).to(device))

                    labels.append(1)
                else:
                    day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x], toCheck])
                    labels.append(0)
                

