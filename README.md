# michinaga

Transformers for trend analysis and binary price prediction. The model uses temporal attention and a LSTM to analyze the direction of a stock. Named for the greatest of the Fujiwara, and largely inspired by the paper cited below.

## Problem

The aim of the preliminary model is to make binary predictions about stocks, based on Tweet information along with raw price data. I employ the use of a 5 day lag period to make a prediction about a target day. The dataset consists of Tweets from 2014 to 2016, along with the price data from said period.

## Model

![alt text](https://github.com/Lysander-curiosum/michinaga/blob/main/model.png?raw=true)


Accuracy so far, on the binary price prediction from the 5-day lag coupled with the average tweet data:
71.76% 

```
@article{Expert Systems with Applications,
    title   = {Transformer-based network for stock movement prediction},
    author  = {Qiuyue Zhang, Chao Qin, Yufeng Zhang, Fangxun Bao, Caiming Zhang, Peide Liu},
    journal = {Elsevier},
    year    = {2022},
    volume  = {212}
}
```
