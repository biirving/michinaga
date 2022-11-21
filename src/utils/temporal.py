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
    def __init__(self, dim):
        super(temporal, self).__init__()

        self.d = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

        self.w_d = nn.Linear(dim, dim)
        self.v_info = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

        self.v_inf = None
        self.v_dep = None

        # the d_target is also used in the processing of this value
        self.v_dependency = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
    
        # then the v output is a pointwise multiplication? 
        self.z_aux = nn.Sequential(nn.Linear(dim, 1), nn.Softmax(dim = 0))

        # the final 'binary' prediction as proposed in the teanet paper
        self.z_final = nn.Sequential(nn.Linear(dim, dim), nn.Softmax())


    # debug forward pass
    def forward(self, input):
        d_vals = self.d(input)
        self.v_inf = self.v_info(d_vals)
        self.v_dep = torch.matmul(d_vals[0], torch.transpose(self.v_dependency(d_vals), 0, 1))
        print(self.v_dep.shape)
        print(self.v_inf.shape)
        v = torch.mul(self.v_inf.view(5), self.v_dep)
        auxilary_predictions = self.z_aux(d_vals)
        print('auxiliary predictions shape', auxilary_predictions.shape)
        print('auxilary predictions', auxilary_predictions)
        print('v shape', v.shape)
        print('d_vals shape', d_vals[len(d_vals)-1].shape)
        print(torch.matmul(auxilary_predictions, v).shape)
        final = self.z_final(torch.cat((torch.matmul(torch.transpose(auxilary_predictions, 0, 1), v), d_vals[len(d_vals) - 1])))
        return final, auxilary_predictions
