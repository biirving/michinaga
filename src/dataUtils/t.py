import torch
import ta
from ta.momentum import rsi
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt



# is it from back to front? front to back?
data = torch.load('/home/benjamin/Desktop/ml/michinaga/src/dataUtils/sp500/AMZN.pt')
print(data[:, 3])
print(data[:, 3])

rsi_1 = rsi(pd.DataFrame(torch.flip(data[:, 3], dims = [0]), columns=['close'])['close'], window=14, fillna = False)
print(rsi_1)

def plot(x1, y1):
    plt.plot(x1, y1, label = 'RSI')
    plt.legend()
    plt.show()

plot(np.arange(100), rsi_1[0:100])





