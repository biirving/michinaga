import torch

googl_data = torch.load('sp500/GOOG.pt').numpy().astype('double')
print(googl_data)
print(googl_data.shape)