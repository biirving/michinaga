"""
Have to prepare the dataset for input into the model

Each individual input will have a lag period of 5, with a 4 tweets associated with 
trading day.

We want to return a list of all of the prepared inputs to the model. We also 
have to 'label' the examples, making the y values for each associated x
value. 


It would be ideal to use the Twitter api in order to grab live data <-- would result in a more interesting problem

"""

# Training data
import json
import wordEmbedding
import torch
from sys import platform
import os
from os.path import exists
import numpy as np

# will this import be fucked? 
from wordEmbedding import wordEmbedding

# How this file is run will depend on the device.
# The filepath is also specific to your computer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataPrep:
    def __init__(self, lag_period, movement_ratio_type, embedding, mode, stacked):
        #self.price_data = []
        #self.price_dates = []
        #self.tweet_data = []  

        self.x_data = []
        self.y_data = []
        self.lag_period = lag_period
        self.movement_ratio_type = movement_ratio_type
        # uh oh
        self.wordembedder = wordEmbedding(embedding, mode, stacked)

    # create a multidimensional tensor from a list of tensors
    def createTensor(self, tensor, dim):
        counter = 0
        for t in tensor:
            if(counter == 0):
                toReturn = t.view(1, dim)
            else:
                toReturn = torch.cat((toReturn, t.view(1, dim)), 0)
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
            print(counter)
            ticker = f[63:]
            tickername = ticker.split('.')[0]
            open_file = open(f)
            price_file = open_file.readlines()

            # check if there are even enough tweet days for one data point
            if platform == "darwin":
                dir_path = r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(tickername)
            elif platform == "win64":
                dir_path = r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(tickername)
            
            # check if there are even any tweets for the given ticker
            if(not(os.path.exists(dir_path))):
                continue

            num_tweets = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

            if(num_tweets < 5):
                # we skip this ticker, because there does not exist viable data
                continue

            # we want to accumulate a datapoint with 5 days, and a tweet vector 
            # for each of those five days
            # if each tweet is an average, that means each day only has ONE tweet vector associated with it
            # which means that we can process more quickly through the forward pass of the teanet model
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

                #print('y', y)
                # we build the price data only adding a 'value' if the day has a corresponding tweet vector
                # we go from back to front in terms of moving through the file
                tweets_checked = 0

                while(len(x_vals) < 5 and tweets_checked <= num_tweets and y > 0):
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

                    # if the corresponding date has tweet data for it
                    if(exists):
                        tweets_checked += 1
                        movement_ratios.append(prices[1])
                        price_values_indices.append(y)
                        if platform == "darwin":
                            tweet_file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(tickername) + '/' + str(date))
                        elif platform == "win64":
                            tweet_file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(tickername) + '\'' + str(date))

                        # the tweets from this specific date
                        text = tweet_file.readlines()
                        # we are going to be doing an average, for some days only have 1 tweet. I don't want dynamic shaping in the model itself <-- could mess with the weights in a strange way
                        for t in range(len(text)):
                            tweet_dict = json.loads(text[t])
                            input = tweet_dict['text']
                            embedded_tweet = self.wordembedder.embed(input)
                            if(t == 0):
                                tweets = embedded_tweet
                            else:
                                tweets += embedded_tweet
                        tweets /= len(tweets)
                        x_vals.append([tweets, torch.tensor([float(x) for x in prices[2:6]]).to(device)])
                        tweet_vals.append(tweets)
                        price_vectors.append(torch.tensor([float(x) for x in prices[2:6]]).to(device))
                        y -= 1
                        tweet_file.close() 
                    else:
                        # we do not consider the price date if it has no corresponding date 
                        y -= 1
                # now we have to determine if the x sample that we have accumulated is a positive or negative sample
                if(len(x_vals) == self.lag_period):
                    # we actually want this to coincide with the value that lies just
                    # beyond the lag period, on the 6th day. If this is somewhat positive, we add it to the dataset
                    movement_ratio = float(price_file[y].split()[1])
                    #movement_ratio = float(movement_ratios[len(movement_ratios) - 1])
                    # in the original paper, they only appended the data point to the list if the movement ratio fell beyond a certain threshold
                    if(movement_ratio <= -0.005 or movement_ratio >= 0.005):
                        #self.x_data.append(x_vals)
                        self.x_data.append([self.createTensor(tweet_vals, 100), self.createTensor(price_vectors, 4)])
                        # here we should store the corresponding ticker along with the date in a tuple form
                        if(movement_ratio >= 0.005):
                            self.y_data.append(torch.tensor([1, 0]).to(device))
                        else:
                            self.y_data.append(torch.tensor([0, 1]).to(device))
                    # we set the new price value indice to one to the left of the previous
                    x = price_values_indices[1]
            break
        return self.x_data, self.y_data


        

okay = dataPrep(5, 'last', 'twitter', 'average', False)
x_data, y_data = okay.returnData()
torch.save(x_data, 'x_data.pt')
torch.save(y_data, 'y_data.pt')








        

