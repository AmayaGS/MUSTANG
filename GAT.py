# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:22:22 2023

@author: AmayaGS
"""

import torch
import torch.nn.functional as F
#from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling
from SAGPool import SAGPool


class GAT(torch.nn.Module):
    
    """Graph Attention Network"""
    
    def __init__(self, dim_in, dim_h, dim_out, heads=8, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_h, heads=1)
        self.pool1 = SAGPool(dim_h, ratio=self.pooling_ratio)
        
        self.lin1 = torch.nn.Linear(dim_h * 2, 3)
        #self.lin2 = torch.nn.Linear(dim_h, dim_h // 2)
        #self.lin3 = torch.nn.Linear(dim_h // 2, 3)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.selu(h)
        
        x_pool, edge_index_pool, _, batch, _ = self.pool1(h, edge_index, None, batch)
        x_pool = torch.cat([gmp(x_pool, batch), gap(x_pool, batch)], dim=1)
        x_pool = F.dropout(x_pool, p=0.4, training=self.training)
        
        x_out = F.relu(self.lin1(x_pool))
        x_out = F.log_softmax(x_out, dim=-1)

        return x_out
    
#%%


class GAT_topK(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=8, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GATv2Conv(dim_in, 512, heads=heads)
        self.gat2 = GATv2Conv(512*heads, 256, heads=4)
        self.gat3 = GATv2Conv(256*4, 128, heads=2)
        self.gat4 = GATv2Conv(128*2, 64, heads=1)
                
        self.topk1 = TopKPooling(4096, pooling_ratio)
        self.topk2 = TopKPooling(256*4, pooling_ratio)
        self.topk3 = TopKPooling(128*2, pooling_ratio)
        self.topk4 = TopKPooling(64, pooling_ratio)
        
        self.lin1 = torch.nn.Linear(38, 19)
        self.lin2 = torch.nn.Linear(19, 8)
        self.lin3 = torch.nn.Linear(8, 3)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat4(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.lin1(x)
        x = F.selu(x)
        x = self.lin2(x)
        x = F.selu(x)
        x = self.lin3(x)
        x = F.selu(x)
        x_out = F.log_softmax(x, dim=-1)

        return x



# %%

# import torch
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GraphConv, TopKPooling
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# import torch.nn.functional as F
# from layers import SAGPool



# class Net(torch.nn.Module):
#     def __init__(self,args):
#         super(Net, self).__init__()
#         self.args = args
#         self.num_features = args.num_features
#         self.nhid = args.nhid
#         self.num_classes = args.num_classes
#         self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout_ratio
        
#         self.conv1 = GCNConv(self.num_features, self.nhid)
#         self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
#         self.conv2 = GCNConv(self.nhid, self.nhid)
#         self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
#         self.conv3 = GCNConv(self.nhid, self.nhid)
#         self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

#         self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
#         self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
#         self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv3(x, edge_index))
#         x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = x1 + x2 + x3

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = F.log_softmax(self.lin3(x), dim=-1)

#         return x
