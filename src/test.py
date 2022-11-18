
import json
from michinaga.src import wordEmbedding, teanet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\price\preprocessed\AAPL.txt', 'r')
price = file.readlines()

prices = price[918:923]
price_values = []
for price in prices[::-1]:
    otay = price.split()
    nums = otay[1:6]
    price_values.append([float(x) for x in nums])

embedder = wordEmbedding('twitter', 'average', False)

tweet_inputs = torch.tensor([])

# we need to generalize this to create the dataset
# also, for each 'example', we need to specify if its positive or negative example (comparing first and last price days?)
for y in range(6, 11):
    if(y < 10):
        file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\tweet\preprocessed\AAPL\2014-01-0' + str(y), 'r')
    else:
        file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\tweet\preprocessed\AAPL\2014-01-' + str(y), 'r')
    text = file.readlines()
    for t in range(4):
        tweet_dict = json.loads(text[t])
        input = tweet_dict['text']
        embedded_tweet = embedder.embed(input)
        if(t == 0):
            tweets = embedded_tweet.view(1, 100)
        else:
            tweets = torch.cat((tweets, embedded_tweet.view(1, 100)), 0)
    if(y == 6):
        tweet_inputs = tweets.view(1, 4, 100)
    else:
        tweet_inputs = torch.cat((tweet_inputs, tweets.view(1, 4, 100)), 0)

print('tweet input', tweet_inputs.shape)

test = teanet(5, 100, 2, 1, 4, 5)
test.to(device)

print(test.forward([tweet_inputs, price_values]))

