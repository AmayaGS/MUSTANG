# -*- coding: utf-8 -*-
"""
Created on Fri May  5 01:26:26 2023

@author: AmayaGS
"""

import torch
from torch.autograd import profiler

# Define your function to be profiled
def my_function():
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.matmul(x, y)
    return z

# Start profiling
with profiler.profile(use_cuda=True) as prof:
    my_function()

# Print the profiling results
print(prof.key_averages().table(sort_by="cuda_time_total"))
