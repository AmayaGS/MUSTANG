# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:54:53 2023

@author: AmayaGS

"""

import torch 
from torchsummary import summary
from Graph_model import GAT_SAGPool_flops
from torchsummary import summary
from ptflops import get_model_complexity_info

# %%

graph_net = GAT_SAGPool_flops(1024)
graph_net.cuda()

# %%

flops, params = get_model_complexity_info(graph_net, (2000, 1024), as_strings=False, print_per_layer_stat=True, verbose=True)

print("FLOPs:", flops, "Parameters:", params)

# %%

get_model_complexity_info(graph_net, (2000, 1024))


# %%





# $ from fvcore.nn import FlopCountAnalysis
# $ flops = FlopCountAnalysis(model, input)
# $ flops.total()
# 274656
# $ flops.by_operator()
# Counter({'conv': 194616, 'addmm': 80040})
# $ flops.by_module()
# Counter({'': 274656, 'conv1': 48600,
#          'conv2': 146016, 'fc1': 69120,
#          'fc2': 10080, 'fc3': 840})
# $ flops.by_module_and_operator()
# {'': Counter({'conv': 194616, 'addmm': 80040}),
#  'conv1': Counter({'conv': 48600}),
#  'conv2': Counter({'conv': 146016}),
#  'fc1': Counter({'addmm': 69120}),
#  'fc2': Counter({'addmm': 10080}),
#  'fc3': Counter({'addmm': 840})}



# import torchvision.models as models
# import torch
# from deepspeed.profiling.flops_profiler import get_model_profile
# from deepspeed.accelerator import get_accelerator

# with get_accelerator().device(0):
#     model = models.alexnet()
#     batch_size = 256
#     flops, macs, params = get_model_profile(model=model, # model
#                                     input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
#                                     args=None, # list of positional arguments to the model.
#                                     kwargs=None, # dictionary of keyword arguments to the model.
#                                     print_profile=True, # prints the model graph with the measured profile attached to each module
#                                     detailed=True, # print the detailed profile
#                                     module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
#                                     top_modules=1, # the number of top modules to print aggregated profile
#                                     warm_up=10, # the number of warm-ups before measuring the time of each module
#                                     as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
#                                     output_file=None, # path to the output file. If None, the profiler prints to stdout.
#                                     ignore_modules=None) # the list of modules to ignore in the profiling 
# 	
# 	
# import torch
# from torch_geometric.nn import MessagePassing

# # Define your PyTorch Geometric model
# class MyModel(MessagePassing):
#     def __init__(self, ...):
#         ...

# model = MyModel(...)

# # Create a dummy input graph to pass through the model
# input_graph = ... # create a PyTorch Geometric graph

# # Compute the FLOPs for the model
# input_shape = (1, ) + graph_model.x.shape
# flops, _ = torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU], 
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logdir'),
#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=5),
#     with_stack=True, 
#     profile_memory=False)(graph_net, (graph_model, ), input_size=input_shape)
    
# print("FLOPs:", flops/1e9, "Billion")