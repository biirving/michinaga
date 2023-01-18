import torch
from src import teanet
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data = [[[torch.randn(98, 100), torch.randn(3, 100), torch.randn(1, 100), torch.randn(33, 100), torch.randn(7, 100)],
[torch.randn(12, 100), torch.randn(55, 100), torch.randn(4, 100), torch.randn( 9, 100), torch.randn(1, 100)]], torch.randn(2, 5, 4).to(device)]
model = teanet(5, 100, 2, 2, 5, 5, 5)

start = time.clock()
model.foward(x_data)
end = time.clock()

print("inference time: ", end - start)


