"""
Have to prepare the dataset for input into the model

Each individual input will have a lag period of 5, with a 4 tweets associated with 
trading day.

We want to return a list of all of the prepared inputs to the model. We also 
have to 'label' the examples, making the y values for each associated x
value. 


It would be ideal to use the Twitter api in order to grab live data <-- would result in a more interesting problem

"""



"""
In this dataprep_alt branch, I will distill the tweets into a list of lists. 
(This will make batch processing lower, but will lead to the model's ability to handle all of the tweet information)
An adaptive pooling layer will prove different to the averaging methods I have previously employed, for this layer will
benefit from back-prop.

Also some of the papers accounted for multiple symbols present in a phrase, and thus used a bidirectional 
GRU (general reccurrent unit) to create informational dependencies surrounding the said symbol. 
"""

# Training data
import json
import wordEmbedding
import torch
from sys import platform
import os
from os.path import exists
import numpy as np

from wordEmbedding import wordEmbedding

# How this file is run will depend on the device.
# The filepath is also specific to your computer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataPrep:
    def __init__(self, lag_period, movement_ratio_type, embedding, mode, stacked):
        #self.price_data = []
        #self.price_dates = []
        #self.tweet_data = []  
        self.tweet_data = []
        self.price_data = None
        self.x_data = []
        self.y_data = []
        self.lag_period = lag_period
        self.movement_ratio_type = movement_ratio_type
        # uh oh
        self.wordembedder = wordEmbedding(embedding, mode, stacked)

    # create a multidimensional tensor from a list of tensors
    def createTensor(self, tensor, dim):
        toReturn = None
        counter = 0
        for t in tensor:
            if(counter == 0):
                # how to process a varying number of tweets
                toReturn = t.view(1, 1, dim)
                #tensor_list.append(toReturn)
            else:
                toReturn = torch.cat((toReturn, t.view(1, 1, dim)), 0)
                #tensor_list.append(t.view(1, t.shape[0], dim))
            counter += 1
        return toReturn


    # I think that they have to be processed together
    # we have to ensure that the data is such that the tweets correspond with the price data
    # therefore, we should only use price data that has a tweet to go along with it

    def returnData(self):
        # iterating through the price data
        # Do we want to iterate through each ticker in the model (each file in the folder)
        if platform == "darwin":
            #file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/prices/AAPL.txt', 'r')
            directory = r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/prices'
        elif platform == "win64":
            #file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\price\preprocessed\AAPL.txt', 'r')
            directory = r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\price'
        elif platform == 'linux' or platform == 'linux2':
            directory = r'/home/benjamin/Desktop/ml/michinaga/src/data/prices'

        filenames = []
        # iterate over files in
        # that directory
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                filenames.append(f)
            
        # sorted list of all of the price data
        filenames = sorted(filenames)

        # compiling the x price data
        counter = 0
        for f in filenames:
            counter += 1
            # different ticker values on windows and mac
            ticker = f[63:]
            #ticker = f[52:]
            tickername = ticker.split('.')[0]
            open_file = open(f)
            price_file = open_file.readlines()

            # check if there are even enough tweet days for one data point
            if platform == "darwin":
                dir_path = r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(tickername)
            elif platform == "win64":
                dir_path = r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(tickername)
            elif platform == 'linux' or platform == 'linux2':
                dir_path = r'/home/benjamin/Desktop/ml/michinaga/src/data/preprocessed/' + str(tickername)
            
            # check if there are even any tweets for the given ticker
            if(not(os.path.exists(dir_path))):
                continue

            num_tweets = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

            if(num_tweets < self.lag_period):
                # we skip this ticker, because there does not exist viable data
                continue

            
            for x in range(len(price_file) - 1, self.lag_period + 1, - 1):

                # x_vals should be a tensor
                x_vals = []
                tweet_vals = []
                price_vectors = []
                movement_ratios = []
                # here are the indices for the price values, so that we can move through the for loop effectively
                price_values_indices = []

                # the first day in the lag period
                y = x

                # we build the price data only adding a 'value' if the day has a corresponding tweet vector
                # we go from back to front in terms of moving through the file
                tweets_checked = 0

                while(len(x_vals) < self.lag_period and tweets_checked <= num_tweets and y > 0):
                    price_to_consider = price_file[y]
                    prices = price_to_consider.split()
                    # here we check if the price day has a corresponding tweet
                    date = prices[0]
                    exists = False
                    if platform == "darwin":
                        #file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/prices/AAPL.txt', 'r')
                        exists = os.path.isfile(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(tickername) + '/' + str(date))
                        #file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(ticker) + '/' + str(date)
                    elif platform == "win64":
                        #file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\price\preprocessed\AAPL.txt', 'r')
                        exists = os.path.isfile(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(tickername) + '\'' + str(date))
                    elif platform == 'linux' or platform == 'linux2':
                        exists = os.path.isfile(r'/home/benjamin/Desktop/ml/michinaga/src/data/preprocessed/' + str(tickername) + '/' + str(date))
                    # if the corresponding date has tweet data for it

                    if(exists):
                        tweets_checked += 1
                        movement_ratios.append(prices[5])
                        price_values_indices.append(y)
                        if platform == "darwin":
                            tweet_file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(tickername) + '/' + str(date))
                        elif platform == "win64":
                            tweet_file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(tickername) + '\'' + str(date))
                        elif platform == 'linux' or platform == 'linux2':
                            tweet_file = open(r'/home/benjamin/Desktop/ml/michinaga/src/data/preprocessed/' + str(tickername) + '/' + str(date))

                        text = tweet_file.readlines()
                        for t in range(len(text)):
                            tweet_dict = json.loads(text[t])
                            input = tweet_dict['text']
                            embedded_tweet = self.wordembedder.embed(input)
                            if(t == 0):
                                tweets = embedded_tweet.view(1, 100)
                            else:
                                #tweets = torch.cat((tweets, embedded_tweet.view(1, 100)), dim = 0)
                                tweets += embedded_tweet
                        # taking the average of the tweets
                        #tweets /= len(tweets)
                        # In this alternative data consolidation, I will instead append the varying number of tweets
                        # in a simple list format

                        x_vals.append([tweets, torch.tensor([float(x) for x in prices[1:5]]).to(device)])
                        tweet_vals.append(tweets)
                        price_vectors.append(torch.tensor([float(x) for x in prices[1:5]]).to(device))
                        y -= 1
                        tweet_file.close() 
                    else:
                        # we do not consider the price date if it has no corresponding date 
                        y -= 1
                # now we have to determine if the x sample that we have accumulated is a positive or negative sample
                if(len(x_vals) == self.lag_period):
                    movement_ratio = float(price_file[y].split()[5])
                    #movement_ratio = float(movement_ratios[len(movement_ratios) - 1])
                    # in the original paper, they only appended the data point to the list if the movement ratio fell beyond a certain threshold
                    if(movement_ratio <= -0.005 or movement_ratio > 0.0055):
                        # self.x_data.append(x_vals)
                        # should this instead be a tensor of tensors
                        #weets = self.createTensor(tweet_vals, 100)
                        rices = self.createTensor(price_vectors, 4)
                        self.tweet_data.append(tweet_vals)
                        #self.price_data.append(price_vectors)
                        
                        if(self.price_data == None):
                            #self.tweet_data = weets.view(1, self.lag_period, 100)
                            self.price_data = rices.view(1, self.lag_period, 4)
                        else:
                            #self.tweet_data = torch.cat((self.tweet_data, weets.view(1, self.lag_period, 100)), 0)
                            self.price_data = torch.cat((self.price_data, rices.view(1, self.lag_period, 4)))
                        
                        #self.x_data.append(self.createTensor(price_vectors, 4))
                        # here we should store the corresponding ticker along with the date in a tuple form
                        if(movement_ratio > 0.0055):
                            self.y_data.append(torch.tensor([1, 0]).to(device))
                        else:
                            self.y_data.append(torch.tensor([0, 1]).to(device))
                    # we set the new price value indice to one to the left of the previous, where to
                    # begin our next value from
                    x = price_values_indices[1]
        return np.array(self.tweet_data), self.price_data, self.createTensor(self.y_data, 2)
