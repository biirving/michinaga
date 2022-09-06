"""
The TEAnet model, for stock market analysis.


There are two primary stages to the mechnaism that I wish to focus on:

The buy mechanism
    - In the original TEAnet paper, they focus on a binary increase/no increase model
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


class textEncoder(nn.Module):
    def __init__(self, num_heads, dim, n) -> None:
        super().__init__()
        self.attention = classicAttention(num_heads, dim, n)

# eventually package structure will be fixed
class teanet(nn.Module):
    def __init__(self, num_heads, dim, height, width, patch_res = 16) -> None:
        super().__init__()
        self.n = int((height * width) / (patch_res ** 2))
        self.textEncoder = textEncoder(num_heads, dim, self.n)

        


