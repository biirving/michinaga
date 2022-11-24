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

"""
TEANET:
An implementation of a model meant to measure stocks, based around the TEANET model
"""

class textEncoder(nn.Module):
    def __init__(self, num_heads, dim, batch_size) -> None:
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


# I am going to implement first for a batch size of 1

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
    - lag 
        How many prior trading days are being considered with each input
    - tweets 
        Tweet embeddings for all of the trading days in the lag period
    - prices
        normalized prices for all of the trading days in the lag period
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class teanet(nn.Module):
    def __init__(self, num_heads, dim, num_classes, batch_size, k, lag) -> None:
        super().__init__()
        self.dim = dim
        self.k = k

        # instead of the sequential shite
        #self.u = nn.Sequential(nn.Linear(k, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Softmax(dim))
        self.w_u = nn.Parameter(torch.randn(self.dim, self.dim))
        self.w_m = nn.Parameter(torch.randn(self.dim))
        self.textSoftmax = nn.Softmax(dim = 0)
        self.num_classes = num_classes

        # is this processed with a class token, to capture the information of the input?
        #self.classtoken = nn.Parameter(torch.randn())

        self.pos_embed = nn.Parameter(torch.randn(lag, k, dim))
        self.lag = lag

        # I am going to begin with batch size of 1. 
        # so, for the text encoder, I am going to process a single input at a time
        self.batch_size = lag

        self.textEncoder = textEncoder(num_heads, dim, batch_size)


        self.lstm = nn.LSTM(input_size = 9, hidden_size = 5)

        # the dimension of the temporal atention mechanism is the lstm_input size concatenated
        # with the lstm_outpus
        self.temporal = temporal(2 * lag + k, num_classes)

    def forward(self, input):
        counter = 0
        input[0] += self.pos_embed
        # each 'tweet' in this for each loop represents all of the tweets for the TDth trading day
        # should not do this with a forloop
        # should be done in one matrix multiplication
        # but for that, we need to change text encoder 
        for tweet in input[0]:
            # so each forward pass is processing a 'batch' of tweets
            # one more 'zoom' out? 

            # so, in this case, if we use the averaging technique, they can all be processed at once
            m = self.textEncoder.forward(tweet)

            # the tanh operation should be computed separately, because W_m is stored 
            # as a parameter
            inter = self.textSoftmax(torch.matmul(torch.transpose(self.w_u, 0, 1), torch.tanh(self.w_m)))
            output = torch.matmul(m, inter)
            tooAdd = torch.cat((output, input[1][counter]))
            if(counter == 0):
                lstm_in = tooAdd.view(1, 9)
            else:
                lstm_in = torch.cat((lstm_in, tooAdd.view(1, 9)), 0)
            counter += 1
        
        #print('lstm_in', lstm_in)

        # process the output through an lstm
        out = self.lstm(lstm_in)

        # identifying viability of the lstm inputs/outputs
        print('lstm out', out)
        print('lstm_out', out[0].shape)

        # the next step is to feed the concated lstm_in and out into temporal attention
        final, auxilary = self.temporal.forward(torch.cat((lstm_in, out[0]), 1))
        return final, auxilary


