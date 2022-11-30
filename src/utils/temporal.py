"""
Temporal Attention for the TEANet model
"""

import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os

class temporal(nn.Module):
    def __init__(self, dim, num_classes):
        super(temporal, self).__init__()
        self.num_classes = num_classes

        self.d = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

        self.v_info = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

        # the d_target is also used in the processing of this value
        self.v_dependency = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
    
        # then the v output is a pointwise multiplication? 
        self.z_aux = nn.Sequential(nn.Linear(dim, 1), nn.Softmax(dim = 0))

        # the final 'binary' prediction as proposed in the teanet paper
        self.z_final = nn.Sequential(nn.Linear(dim + 5, self.num_classes), nn.Softmax())

        # this should not be necessary, the tensor operations are fucked.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # debug forward pass
    def forward(self, input):
        d_vals = self.d(input)
        v_inf = self.v_info(d_vals)
        v_dep = torch.matmul(d_vals[:, 4, :], torch.transpose(self.v_dependency(d_vals), 1, 2))
        v = torch.mul(v_inf.view(5, 5), v_dep)
        auxilary_predictions = self.z_aux(d_vals)
        final = self.z_final(torch.cat((torch.matmul(torch.transpose(auxilary_predictions, 1, 2), v), d_vals[:, 4, :].view(5, 1, 109)), 2))
        return final, auxilary_predictions
