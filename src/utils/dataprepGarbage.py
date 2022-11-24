
# throwaway stuff from dataprep iteration.

"""


            for x in range(len(price_file) - 1, self.lag_period + 1, - 1):
                

                prices = price_file[x - self.lag_period - 1:x]

                price_values = []
                dates = []

                # we then have to determine if the sample is a 1 or a 0, an increase or a decrease
                prices_without_last = prices[1:]
                movement_ratio_average = 0


                # this would have to be a while loop
                for price in prices_without_last[::-1]:
                    price_parsed = price.split()
                    nums = price_parsed[2:6]

                    # so we should check for the corresponding tweet here 
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
                # instead, we should just create an average tweet over all of the available tweets 9
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
"""