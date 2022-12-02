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
    def __init__(self, dim, num_classes, batch_size):
        super(temporal, self).__init__()
        self.dim = dim
        self.batch_size = batch_size

        self.num_classes = num_classes

        self.d = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

        self.v_info = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

        # the d_target is also used in the processing of this value
        self.v_dependency = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
    
        # then the v output is a pointwise multiplication? 
        self.z_aux = nn.Sequential(nn.Linear(dim, 1), nn.Softmax(dim = 1))

        # the final 'binary' prediction as proposed in the teanet paper
        self.z_final = nn.Sequential(nn.Linear(dim + 1, self.num_classes), nn.Softmax(dim = num_classes))

    """
    Setting the batch size to allow for different sized inputs
    """
    def setBatchSize(self, new):
        self.batch_size = new

    # debug forward pass, use intermediate variables
    def forward(self, input):
        d_vals = self.d(input)
        v_inf = self.v_info(d_vals)
        d_target = d_vals[:, 4, :].view(self.batch_size, 1, self.dim)
        v_dep = torch.matmul(d_target, torch.transpose(self.v_dependency(d_vals), 1, 2))
        v = torch.mul(v_inf.view(self.batch_size, 1, 5), v_dep)
        auxilary_predictions = self.z_aux(d_vals)
        final = self.z_final(torch.cat(
            (torch.matmul(torch.transpose(auxilary_predictions, 1, 2), v.view(self.batch_size, 5, 1)), d_target), 2))
        return final 
