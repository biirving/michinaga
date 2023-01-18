

"""
Class to deal with the pooling problem (differing amounts of tweets for various days)
"""

import torch
from torch import nn, tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# will we have a device setting here? to ensure that the data is being processed? (why is this bad practice)

class pooling(nn.Module):
    """A pooling class, so that the forward pass in this variation of the teanet model is lest complex
        Also, perhaps it can be trained separately?
    """
    def __init__(self, dim, lag):
        super(pooling, self).__init__()
        self.dim = dim
        self.lag = lag
        # multiple pooling layers? A feed forward neural network?
        self.adaptive_pooling = nn.AdaptiveMaxPool2d((1, dim))

    def forward(self, input):
        batch_of_tweets = None
        for x_val in input[0]:
            processed_tweets = None
            # iterate through the days in the x_value
            for day in x_val:
                processed = self.adaptive_pooling(day.view(1, day.shape[0], day.shape[1]))
                if(processed_tweets == None):
                    processed_tweets = processed
                else:
                    processed_tweets = torch.cat((processed_tweets, processed), 1)
            if(batch_of_tweets == None):
                batch_of_tweets = processed_tweets.view(1, self.lag, self.dim)
            else:
                batch_of_tweets = torch.cat((batch_of_tweets, processed_tweets.view(1, self.lag, self.dim)), 0)
        return batch_of_tweets.to(device)