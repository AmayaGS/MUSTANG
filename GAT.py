# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:22:22 2023

@author: AmayaGS
"""

import torch
import torch.nn.functional as F
#from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        h = global_mean_pool(h, 4)
        
        return h, F.log_softmax(h, dim=1)