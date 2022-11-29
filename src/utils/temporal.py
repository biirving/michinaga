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
        self.z_final = nn.Sequential(nn.Linear(dim + 1, self.num_classes), nn.Softmax())

        # this should not be necessary, the tensor operations are fucked.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # debug forward pass
    def forward(self, input):
        print('input', input.shape)
        d_vals = self.d(input)
        print('d shape', d_vals.shape)
        v_inf = self.v_info(d_vals)
        # examine these operations closely
        print(d_vals[:, 4, :].shape)
        print(torch.transpose(self.v_dependency(d_vals), 1, 2).shape)
        v_dep = torch.matmul(d_vals[:, 4, :], torch.transpose(self.v_dependency(d_vals), 1, 2))
        print(v_inf.shape)
        print(v_dep.shape)
        v = torch.mul(v_inf.view(5, 5), v_dep)
        print(v.shape)
        auxilary_predictions = self.z_aux(d_vals)
      #  print('v_dep shape', v_dep.shape)
       # print('v_innf shape', v_inf.shape)
        #print('auxiliary predictions shape', torch.transpose(auxilary_predictions, 0 , 1).shape)
       # print('auxilary predictions', auxilary_predictions)
     #   print('v shape', v.shape)
     #   print('d_vals shape', d_vals[len(d_vals)-1].shape)
     #   print(torch.transpose(auxilary_predictions, 0, 1)[0])
      #  print(v)
       # print(torch.matmul(torch.transpose(auxilary_predictions, 0, 1)[0], torch.transpose(v, -1,0)))
        #print(torch.matmul(auxilary_predictions, v))

        # so why is this not automatically on the device
        final = self.z_final(torch.cat((torch.tensor([torch.matmul(torch.transpose(auxilary_predictions, 0, 1)[0], v)]).to(self.device), d_vals[len(d_vals) - 1])))
        return final, auxilary_predictions
