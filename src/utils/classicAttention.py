import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange
import math


class classicAttention(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, droput=0.):
        super(classicAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads)
        
    def forward(self, input):
        # q, k, v matrices
        q_mat = rearrange(self.q(input), 'b l (h d) -> b l h d', h = self.num_heads)
        v_mat = rearrange(self.k(input), 'b l (h d) -> b l h d', h = self.num_heads)
        k_mat = rearrange(self.v(input), 'b l (h d) -> b l h d', h = self.num_heads)

        # Softmax step, calculated for each row of each head
        inter = self.softmax(torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))

        # dropout
        inter = self.dropout(inter)
        # prepare the vector for input
        final = rearrange(torch.matmul(inter, v_mat), 'b l h d -> b l (h d)', h = self.num_heads)

        # final computation
        return self.multi_mad(final)


