"""
The TEAnet model, for stock market analysis.


There are two primary stages to the mechnaism that I wish to focus on:

The buy mechanism
    - In the original TEAnet paper, they focus on a binary increase/decrease model
    - Instead, I want to focus on trends 
    - Text from the tweets appeared to be as if not more significant of an indicator 
        of a positive/negative price outlook. Will this be the same case for trend identification?
    - if the model predicts that the stock price will fall, the trader will buy the stock when it is possible to buy it at less than 1% 
        of the short price

The sell mechnanism
    - Original authors employed binary system introduced in another paper. When the stock went above a certain
        threshold of 2%, the sell would be triggered. Otherwise the trader will need to sell the stock at the closing price at the end of the day
    - the sell mechanism in this model will focus on trends, similarly to the buy mechanisms 
    - Bullish vs. Bearish
    - The temporal data becomes far more interesting in this case
    - Problems with this strategy:
        The tweets aren't as effective of a mechnanism over large timescales, so trend prediction becomes less clear
        5 days vs 10 days results in a significant price increase 

"""

from torch import nn, tensor
import torch
from michinaga.src.utils import classicAttention, temporal



class textEncoder(nn.Module):
    def __init__(self, num_heads, dim) -> None:
        super().__init__()
        # the multihead attention mechanism
        self.multiHeadAttention = classicAttention(num_heads, dim)
        # the positional embedding will be initialized as a random parameter (will be updated with the backward call)
        # layer normalization in the text encoder of the model
        # you add and normalize
        self.layernorm = nn.LayerNorm(dim)
        # the feed forward neural network
        self.FFN = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        # this is followed by another attention layer
        # not multihead
        self.attention = classicAttention(1, dim)

    """
    Function for the input into the text encoder. We feed in the tweet information (or other information streams)
    After the forward process executes, then we feed the remainder into the LSTM.
    """
    def forward(self, input):
        inter = self.multiHeadAttention.forward(input)
        inter = self.layernorm(inter + input)
        output = self.FFN(inter)
        return self.attention.forward(output)

"""
teanet 
long range dependencies for trend and price analysis

args:
    - num_heads
        The number of attention heads for the text encoder
    - dim
        The dimension of the message embeddings (what will they be projected into)
    - batch size 
        How many inputs are being processed at once
    - k (will this be a dynamic value?)
        How many messages will be considered for each trading day
        Because of the nature of the data that we are working with, 
        this value will be one for now (the embedded tweets averaged)

    - lag 
        How many prior trading days are being considered with each input
    - tweets 
        Tweet embeddings for all of the trading days in the lag period
    - prices
        normalized prices for all of the trading days in the lag period
"""


class teanet(nn.Module):
    def __init__(self, num_heads, dim, num_classes, batch_size, k, lag) -> None:
        super().__init__()
        self.dim = dim
        # deprecated: we are just processing the tweet embeddings for each
        # day as the average of all the tweets available in the dataset
        # to capture as much information as possible
        #self.k = k
        self.w_u = nn.Parameter(torch.randn(self.dim, self.dim))
        self.w_m = nn.Parameter(torch.randn(self.dim))
        self.textSoftmax = nn.Softmax(dim = 0)
        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(torch.randn(batch_size, lag, dim))
        self.lag = lag
        self.batch_size = batch_size
        self.textEncoder = textEncoder(num_heads, dim)
        self.lstm = nn.LSTM(input_size = 104, hidden_size = 5)
        self.temporal = temporal(109, num_classes, batch_size)

    def forward(self, input):
        counter = 0
        input += self.pos_embed
        lstm_text_input = self.textEncoder.forward(input[0])
        lstm_in = torch.cat((lstm_text_input, input[1]), 2)
        out = self.lstm(lstm_in)
        final, auxilary = self.temporal.forward(torch.cat((lstm_in, out[0]), 2))
        return final, auxilary


