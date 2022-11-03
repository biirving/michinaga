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
from michinaga.src import classicAttention


"Where does the text embedding happen though?"


"""
TEANET:
An implementation of a model meant to measure stocks, based around the TEANET model
"""

class textEncoder(nn.Module):
    def __init__(self, num_heads, dim, n, batch_size) -> None:
        super().__init__()
        self.attention = classicAttention(num_heads, dim, n)

        # the positional embedding will be initialized as a random parameter (will be updated with the backward call)
        self.pos_embed = nn.Parameter(torch.randn(batch_size, self.n + 1, dim))

        # layer normalization in the text encoder of the model
        # you add and normalize
        self.layernorm = nn.LayerNorm(dim)

        # the feed forward neural network
        self.FFN = nn.Sequential(nn.Linear(dim), nn.ReLU(), nn.Linear(dim))

        # this is followed by another attention layer
        # not multihead
        self.attention_output = classicAttention(1, dim, n)

    """
    Function for the input into the text encoder. We feed in the tweet information (or other information streams)
    After the forward process executes, then we feed the remainder into the LSTM.
    """
    def forward(self, input):
        input += self.pos_embed()
        inter = self.attention.forward(input)
        inter = self.layernorm(inter)
        output = self.FFN(inter)
        # return the output for the LSTM input
        return self.attention_output.forward(output)


# how is the price extracted
# closing price, highest price, and the lowest price
# the main blocker for this part of the model is found in how these three separate vectors
# are extracted from the data
# p_td = [p^{c}_td, p^{h}_td, p^{L}_td]
# p_a = (p_td / p^{c}_{td-1})  - 1

# this obviously needs work. How are we going to get the data? How will it be molded into an
# input that we can use

# does this need to be a separate class 
class priceExtractor(nn.Module):
    """In the original TEANet model, the authors use attention to reach back 5 days for each 
    stock analyzed. The purpose of this initial model is to identify trends, specifically stocks 
    that are entering bullish trends."""
    def __init__(self, dim, batch_size, closing_prices, high_prices, low_prices):
        super(priceExtractor, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.closing_prices = closing_prices
        self.high_prices = high_prices
        self.low_prices = low_prices

        self.price_vector = torch.cat(self.closing_prices, self.high_prices, self.low_prices)

        # divide each vector by the closing prices, so the closing prices become a vector of 1s??
        # also this operation will most certainly have to be debugged, because it is dependent on
        # batch_size
        self.price_vector = torch.div(self.price_vector, self.closing_prices) - 1

        
    # the price vector is simply returned for the input into the LSTM
    def forward(self, x):
        return self.price_vector


# eventually package structure will be fixed
class teanet(nn.Module):
    def __init__(self, num_heads, dim, height, width, batch_size) -> None:
        super().__init__()
        self.textEncoder = textEncoder(num_heads, dim, self.n, batch_size)

        # here we will transition into the LSTM portion of the model
        # can be a call to the torch lstm

        # input size?
        # depends on the number of days of stock information that we use
        # if its 5 days, than the vector will be of size dim + 15
        # hidden state size?
        self.lstm = nn.LSTM(input_size = dim + 15, hidden_size = dim + 15)

        # temporal attention




        
        
        


        


