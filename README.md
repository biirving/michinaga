# michinaga

Transformers for trend analysis and binary price prediction. The model uses temporal attention and a LSTM to analyze the direction of a stock. Named for the greatest of the Fujiwara. 

## Problem

The aim of the preliminary model is to make binary predictions about stocks, based on Tweet information along with raw price data. I employ the use of a 5 day lag period to make a prediction about a target day. The dataset consists of Tweets from 2014 to 2016, along with the price data from said period.

## Model

![alt text](https://github.com/Lysander-curiosum/michinaga/blob/main/model.png?raw=true)

Above is the basic architecture of the model. The input consists of the embeddings for the Tweets in the 5 day lag period, which are processed using the FLAIR nlp library. These are fed into a traditional transformer encoder, the outputs of which are then concatenated with the price data. These are in turn processed by an LSTM, then finally fed into a temporal attention mechanism to predict an increase or decrease in the price. For more details, please take a look at the paper attached to the repository. 

Accuracy so far, on the binary price prediction from the 5-day lag coupled with the average tweet data:
71.76% 
