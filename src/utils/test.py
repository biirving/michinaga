import torch

x_tweets = torch.load('x_tweet_data.pt')

torch.save(x_tweets[0:5], 'x_tweet_test.pt')

x_prices = torch.load('x_price_data.pt')

torch.save(x_prices[0:5], 'x_price_test.pt')


