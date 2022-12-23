"""
The TEAnet model, for stock market analysis.
"""

from torch import nn, tensor
import torch
from michinaga.src.utils import classicAttention, temporal
from einops import repeat

class textEncoder(nn.Module):
    def __init__(self, num_heads, dim) -> None:
        super().__init__()
        self.multiHeadAttention = classicAttention(num_heads, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.FFN = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attention = classicAttention(1, dim)

    """
    Function for the input into the text encoder. We feed in the tweet information (or other information streams)
    After the forward process executes, then we feed the remainder into the LSTM.
    """
    def forward(self, input):
        inter = self.multiHeadAttention.forward(input)
        new = self.layernorm(inter + input)
        output = self.FFN(new)
        return self.attention.forward(output)



"""
We want to have layernorms before each segment in the model.
Helps to alleviate vanishing/exploding gradient
"""
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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

    DEPRECATED
    - k 
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


"""
In this alternate version of the model, we are going to use a revamped version of
the previous Tweet collection methodology. I want to use as much information as possible
(namely, all the Tweets in the dataset). 

How does the utility of this extend beyond the experiment? (practical use?)
Data will always be imperfect in the real world. Thus, our model should be able to account
for irregularity in terms of tensor size.

The goal is to use adaptive pooling (and/or padding?) along with making the word embeddings part of the model
itself (so that everything can be trained together?) <-- how do you train word embeddings?

One alternative to this dataprep alteration would be to instead use a determined number of Tweets (compare?)


How I really want to test all of these alternatives: on live fucking data tbh
"""

class teanet(nn.Module):
    def __init__(self, num_heads, dim, num_classes, batch_size, lag, num_encoders, num_lstms, word_embed) -> None:
        super().__init__()

        """
        How can the FLAIR embeddings be trained? As LMs?
        """
        self.word_embed = word_embed

        """
        Experiment:
        max pooling / average pooling?

        The effectiveness of both methods is obviously different than the advantages found in convlutional nn's
        which are specialized for image processing

        We want to project the bank of tweets into one vector for processing by the model
        """
        self.adaptive_pooling = nn.AdaptiveMaxPool2d((1, dim))


        self.dim = dim
        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(torch.randn(1, lag, dim))
        self.price_pos_embed = None
        self.lag = lag
        self.batch_size = batch_size

        """
        Research to implement?
        """
        self.dropout = nn.Dropout(p = 0.)
    
        """
        consider increasing the number of encoder blocks, to stabilize performance.
        Testing with 6 encoders
        """
        self.textEncoder = nn.ModuleList([textEncoder(num_heads, dim), nn.LayerNorm(dim)] * num_encoders)
        self.lstm = nn.LSTM(input_size = 104, hidden_size = 5, num_layers = num_lstms)
        self.temporal = nn.Sequential(nn.LayerNorm(109), temporal(109, num_classes, batch_size, lag))

    """
    This should be dynamic based on the input from the user
    """
    def setBatchSize(self, new):
        self.batch_size = new
        self.temporal[1].setBatchSize(new)


    """
    The forward pass will be different. 

    Input format: list of list of tensors?
    because we will be processing a lag period for whatever the appropriate batch size is determined to be

    Funny, that so much of the model architecture is determined by the shape that the data takes,
    especially in the early stages.
    """
    def forward(self, input):
        # each lag period + prediction
        for x_val in input:
            # iterating over each day in the specific x value
            for day in x_val:
                # wait, so are we training the word embeddings and the
                # adaptive pooling in tandem? 
                # That seems so inefficient (but will be closer to reality) <-- should be a separate compoennt in
                # the pipeline, no?
                # there needs to be another for loop?
                embedded = self.word_embed(day[0])
                



        toFeed = input[0] + repeat(self.pos_embed, 'n d w -> (b n) d w', b = self.batch_size)
        for text in self.textEncoder:
            toFeed = text.forward(toFeed)
        lstm_in = torch.cat((toFeed, input[1]), 2)
        out = self.lstm(lstm_in)
        final = self.temporal.forward(torch.cat((lstm_in, out[0]), 2))
        return final