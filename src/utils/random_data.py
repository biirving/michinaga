

""" 
a file to create the train and test splits
"""

import torch
from michinaga.src import teanet
import random


device = torch.device('cuda')
    
def build_random_sample(indices, tweets, prices, y):
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

def train(model, params):
    pass
    
# for this final data prep we don't need to run this multiple times
if __name__ == "__main__":
    x_price_data = torch.load('/home/benjamin/Desktop/ml/michinagaData/x_price_data.pt')
    x_tweet_data = torch.load('/home/benjamin/Desktop/ml/michinagaData/x_tweet_data.pt')
    y_data = torch.load('/home/benjamin/Desktop/ml/michinagaData/y_data.pt')

    x_train_indices = random.sample(range(len(x_price_data)), int(0.75 * float(len(x_price_data))))
    x_test_indices = []
    for x in range(len(x_price_data)):
        if(not(x in x_train_indices)):
            x_test_indices.append(x)

    x_train_tweets, x_train_prices, y_train = build_random_sample(x_train_indices, x_tweet_data, x_price_data, y_data)
    x_test_tweets, x_test_prices, y_test = build_random_sample(x_test_indices, x_tweet_data, x_price_data, y_data)
    
    torch.save(x_train_tweets, 'x_train_tweets.pt')
    torch.save(x_train_prices, 'x_train_prices.pt')
    torch.save(y_train, 'y_train.pt')
    torch.save(x_test_tweets, 'x_test_tweets.pt')
    torch.save(x_test_prices, 'x_test_prices.pt')
    torch.save(y_test, 'y_test.pt')