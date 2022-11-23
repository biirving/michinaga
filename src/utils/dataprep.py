"""
Have to prepare the dataset for input into the model

Each individual input will have a lag period of 5, with a 4 tweets associated with 
trading day.

We want to return a list of all of the prepared inputs to the model. We also 
have to 'label' the examples, making the y values for each associated x
value. 


It would be ideal to use the Twitter api in order to grab live data <-- would result in a more interesting problem

"""


# you know what you have to do
# god this is boring
# it will be easier with subsequent data

# Training data
import json
import wordEmbedding
import torch
from sys import platform
import os

# will this import be fucked? 
from wordEmbedding import wordEmbedding

# How this file is run will depend on the device.
# The filepath is also specific to your computer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataPrep:
    def __init__(self, lag_period, movement_ratio_type, k, embedding, mode, stacked):
        self.price_data = []
        self.price_dates = []
        self.tweet_data = []  
        self.y_data = []
        self.lag_period = lag_period
        self.movement_ratio_type = movement_ratio_type
        # uh oh
        self.wordembedder = wordEmbedding(embedding, mode, stacked)
        self.k = k

    # we might have to do the price data second, there is more price data available than tweet data obviously 

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
        print(len(filenames))

        # compiling the x price data
        counter = 0
        for f in filenames:
            print(counter)
            counter += 1
            ticker = f[63:]
            open_file = open(f)
            price_file = open_file.readlines()
            for x in range(len(price_file) - 1, self.lag_period + 1, - 1):
                prices = price_file[x - self.lag_period - 1:x]

                price_values = []
                dates = []

                # we then have to determine if the sample is a 1 or a 0, an increase or a decrease
                prices_without_last = prices[1:]
                movement_ratio_average = 0
                for price in prices_without_last[::-1]:
                    price_parsed = price.split()
                    nums = price_parsed[2:6]
                    dates.append(price_parsed[0])
                    price_values.append(torch.tensor([float(x) for x in nums]).to(device))

                    # we observe the last movement ratio? How should we determine positive/negative datapoints? 

                    # this option will greatly depend upon what we are trying to identify, a positive trend, or a binary price
                    # prediction?
                    # for the initial model, which will follow the binary price prediction scheme, we will simply look at the
                    # price movement ratio for the price value beyond the lat period
                    movement_ratio_average += float(price_parsed[1])
                if(self.movement_ratio_type == 'average'):
                    movement_ratio_average += float(prices[0].split()[1])
                    movement_ratio =  movement_ratio_average / (lag_period + 1)
                    # the datapoint for such an option would be completely different - for trend identification, what data would we even
                    # use? <-- more effective for long term investment
                # the option used for our initial run through
                elif(self.movement_ratio_type == 'last'):
                    movement_ratio = float(prices[0].split()[1])
                    # in the original paper, they only appended the data point to the list if the movement ratio fell beyond a certain threshold
                    if(movement_ratio <= -0.005 or movement_ratio >= 0.005):
                        self.price_data.append(price_values)
                        # here we should store the corresponding ticker along with the date in a tuple form
                        self.price_dates.append((dates, ticker))
                        if(movement_ratio >= 0.005):
                            self.y_data.append(torch.tensor([1, 0]).to(device))
                        else:
                            self.y_data.append(torch.tensor([0, 1]).to(device))

                elif(self.movement_ratio_type == 'first_v_last'):
                    first_close = float(prices[len(prices) - 1].split()[6])
                    last_close = float(prices[0].split()[6])
                    movement_ratio = last_close / first_close
            
            # close the price file
            open_file.close()
                
            
        ############################################ Tweet data collection ############################################
        # the tweet data should correspond with each price datapoint
        # we have the dates stored, but we need to store their corresponding ticker
        for date in self.price_dates:
            print(date)
            # then, we gather the corresponding tweet data
            for y in range(len(date)):
                if platform == "darwin":
                    #file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/prices/AAPL.txt', 'r')
                    file = open(r'/Users/benjaminirving/Desktop/mlWalk/michinaga/src/data/preprocessed/' + str(date[1]) + '/' + str(date[0]))
                elif platform == "win64":
                    #file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\price\preprocessed\AAPL.txt', 'r')
                    file = open(r'C:\Users\Benjamin\Desktop\ml\stocknet-dataset\preprocessed\'' + str(date[1]) + '\'' + str(date[0]))

                text = file.readlines()
                for t in range(self.k):
                    tweet_dict = json.loads(text[t])
                    input = tweet_dict['text']
                    embedded_tweet = embedder.embed(input)
                    if(t == 0):
                        tweets = embedded_tweet.view(1, 100)
                    else:
                        tweets = torch.cat((tweets, embedded_tweet.view(1, 100)), 0)
                if(y == 0):
                    tweet_inputs = tweets.view(1, self.k, 100)
                else:
                    tweet_inputs = torch.cat((tweet_inputs, tweets.view(1, self.k, 100)), 0)

            self.tweet_data.apped(tweet_inputs.to(device))

        return self.price_data, self.tweet_data, self.y_data



        

okay = DataPrep(5, 'last', 4, 'twitter', 'average', False)
prices, tweets, y_data = okay.returnData()
print('prices', prices)
print('tweets', tweets)
print('y data', y_data)






        

