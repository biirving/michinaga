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
        self.pos_embed = nn.Parameter(torch.randn(batch_size, dim))

        # layer normalization in the text encoder of the model
        # you add and normalize
        self.layernorm = nn.LayerNorm(dim)

        # the feed forward neural network
        self.FFN = nn.Sequential(nn.Linear(dim), nn.ReLU(), nn.Linear(dim))

        # this is followed by another attention layer
        # not multihead
        self.attention = classicAttention(1, dim)

    """
    Function for the input into the text encoder. We feed in the tweet information (or other information streams)
    After the forward process executes, then we feed the remainder into the LSTM.
    """
    def forward(self, input):
        input += self.pos_embed()
        inter = self.multiHeadAttention.forward(input)
        inter = self.layernorm(inter)
        output = self.FFN(inter)
        # return the output for the LSTM input
        return self.attention.forward(output)


# eventually package structure will be fixed
class teanet(nn.Module):
    def __init__(self, num_heads, dim, height, width, batch_size) -> None:
        super().__init__()
        self.textEncoder = textEncoder(num_heads, dim, self.n, batch_size)
        self.lstm = nn.LSTM(input_size = dim + 15, hidden_size = dim + 15)