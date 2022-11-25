
import torch


x_data = torch.load('x_data.pt')
y_data = torch.load('y_data.pt')
print(len(x_data))
print(len(y_data))
print(y_data)
print(x_data[0])