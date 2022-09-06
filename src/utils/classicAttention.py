import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os 

class classicAttention(nn.Module):
    def __init__(self, num_heads, dim, n):
        super(classicAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n = n
        self.Dh = int(self.dim/self.num_heads)

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * 3 * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        
    def forward(self, input):
        # q, k, v matrices
        q_mat = rearrange(self.q(input), 'b n (h d) -> b h n d', h = self.num_heads)
        v_mat = rearrange(self.k(input), 'b n (h d) -> b h n d', h = self.num_heads)
        k_mat = rearrange(self.v(input), 'b n (h d) -> b h n d', h = self.num_heads)

        # Softmax step, calculated for each row of each head
        inter = self.softmax(torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))

        # prepare the vector for input
        final = rearrange(torch.matmul(inter, v_mat), 'b h n d -> b n (h d)', h = self.num_heads)

        # final computation
        return self.multi_mad(final)