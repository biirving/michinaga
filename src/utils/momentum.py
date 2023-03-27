import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os

# how to deal with this data?


y_val = torch.load('/home/benjamin/Desktop/ml/macd_rsi_y_data.pt')
x_val = torch.load('/home/benjamin/Desktop/ml/macd_rsi_x_data.pt')

"""
MACD BASIC STRATEGY
    Buy: 𝑀𝑎𝑐𝑑𝑡−1 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 > 0)
    Sell: 𝑀𝑎𝑐𝑑𝑡−1 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 < 0)

    MACD --> signal crossover AND (∀{𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 } ≤ 𝐿𝑜𝑤𝑒𝑟 𝑇ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑

    THIS IS A BUY CLASSIFIER!

returns:
    the prepared macd/rsi vector with the following values
    [𝑀𝑎𝑐𝑑𝑡−1, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1, 𝑀𝑎𝑐𝑑𝑡, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡, 𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5]
"""

# lets throw this shit at classic attention
# systematicall, walking down the forest path.

# 1.) Make a huge model

# 2.) split into training and test

# 3.) run a training loop on it to see how it preforms (USE TORCH PARALLEL ON DISCOVERY, ALONG WITH MULTIPLE GPUS)

# 4.) While the training loops run, really look at the structure of the attention code. What should the model look like?
# What loss function works the best in my problem space?


