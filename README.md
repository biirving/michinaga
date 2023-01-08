# michinaga

Transformers for trend analysis and binary price prediction. The model uses temporal attention and a LSTM to analyze the direction of a stock. Named for the greatest of the Fujiwara, and largely inspired by the paper cited below.

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
