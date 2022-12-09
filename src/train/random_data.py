

""" 
a file to create the train and test splits
"""

import torch
import random


device = torch.device('cuda')

"""
This class can be called with the training module, in order to mix up which training data the model is being
trained on
"""
class random_data:
    def __init__(self):
        pass

    def build_random_sample(self, indices, tweets, prices, y):
        x_sampled_tweet_data = None
        x_sample_price_data = None
        y_data = None
        counter = 0
        for index in indices:
            print(counter)
            counter+=1
            if(x_sampled_tweet_data == None):
                x_sampled_tweet_data = tweets[index].view(1, 5, 100).to(device)
                x_sample_price_data = prices[index].view(1, 5, 4).to(device)
                y_data = y[index].view(1, 2).to(device)
            else:
                x_sampled_tweet_data = torch.cat((x_sampled_tweet_data, tweets[index].view(1, 5, 100).to(device)), 0)
                x_sample_price_data = torch.cat((x_sample_price_data, prices[index].view(1, 5, 4).to(device)), 0)
                y_data = torch.cat((y_data, y[index].view(1, 2).to(device)), 0)
        return x_sampled_tweet_data, x_sample_price_data, y_data

        
    # for this final data prep we don't need to run this multiple times
    def forward(self):
        x_price_data = torch.load('x_price_data.pt')
        x_tweet_data = torch.load('x_tweet_data.pt')
        y_data = torch.load('y_data.pt')

        x_train_indices = random.sample(range(len(x_price_data)), int(0.75 * float(len(x_price_data))))
        x_test_indices = []
        for x in range(len(x_price_data)):
            if(not(x in x_train_indices)):
                x_test_indices.append(x)

        x_train_tweets, x_train_prices, y_train = self.build_random_sample(x_train_indices, x_tweet_data, x_price_data, y_data)
        x_test_tweets, x_test_prices, y_test = self.build_random_sample(x_test_indices, x_tweet_data, x_price_data, y_data)
        
        torch.save(x_train_tweets, 'x_train_tweets.pt')
        torch.save(x_train_prices, 'x_train_prices.pt')
        torch.save(y_train, 'y_train.pt')
        torch.save(x_test_tweets, 'x_test_tweets.pt')
        torch.save(x_test_prices, 'x_test_prices.pt')
        torch.save(y_test, 'y_test.pt')