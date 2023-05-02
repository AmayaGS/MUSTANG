# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:22:22 2023

@author: AmayaGS
"""

import torch
import torch.nn.functional as F
#from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling, SAGPooling


# class GAT(torch.nn.Module):
    
#     """Graph Attention Network"""
    
#     def __init__(self, dim_in, dim_h, dim_out, heads=8, pooling_ratio=0.5):
        
#         super().__init__()
        
#         self.pooling_ratio = pooling_ratio
        
#         self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
#         self.gat2 = GATv2Conv(dim_h*heads, dim_h, heads=1)
#         self.pool1 = SAGPool(dim_h, ratio=self.pooling_ratio)
        
#         self.lin1 = torch.nn.Linear(dim_h * 2, 3)
#         #self.lin2 = torch.nn.Linear(dim_h, dim_h // 2)
#         #self.lin3 = torch.nn.Linear(dim_h // 2, 3)

#     def forward(self, data):
        
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         h = self.gat1(x, edge_index)
#         h = F.elu(h)
#         h = F.dropout(h, p=0.4, training=self.training)
#         h = self.gat2(h, edge_index)
#         h = F.selu(h)
        
#         x_pool, edge_index_pool, _, batch, _ = self.pool1(h, edge_index, None, batch)
#         x_pool = torch.cat([gmp(x_pool, batch), gap(x_pool, batch)], dim=1)
#         x_pool = F.dropout(x_pool, p=0.4, training=self.training)
        
#         x_out = F.relu(self.lin1(x_pool))
#         x_out = F.log_softmax(x_out, dim=-1)

#         return x_out
    
#%%


class GAT_SAGPool(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GATv2Conv(dim_in, 512, heads=1)
        self.gat2 = GATv2Conv(512*heads, 512, heads=1)
        self.gat3 = GATv2Conv(512*heads, 512, heads=1)
        self.gat4 = GATv2Conv(512*heads, 512, heads=1)
                
        self.topk1 = SAGPooling(512, pooling_ratio)
        self.topk2 = SAGPooling(512, pooling_ratio)
        self.topk3 = SAGPooling(512, pooling_ratio)
        self.topk4 = SAGPooling(512, pooling_ratio)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1 + x2 + x3 + x4
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out

# %%

class GAT_TopK(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GATv2Conv(dim_in, 512, heads=1)
        self.gat2 = GATv2Conv(512*heads, 512, heads=1)
        self.gat3 = GATv2Conv(512*heads, 512, heads=1)
        self.gat4 = GATv2Conv(512*heads, 512, heads=1)
                
        self.topk1 = TopKPooling(512, pooling_ratio)
        self.topk2 = TopKPooling(512, pooling_ratio)
        self.topk3 = TopKPooling(512, pooling_ratio)
        self.topk4 = TopKPooling(512, pooling_ratio)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1 + x2 + x3 + x4
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out


# %%

class GCN_SAGPool(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GCNConv(dim_in, 512)
        self.gat2 = GCNConv(512*heads, 512)
        self.gat3 = GCNConv(512*heads, 512)
        self.gat4 = GCNConv(512*heads, 512)
                
        self.topk1 = SAGPooling(512, pooling_ratio)
        self.topk2 = SAGPooling(512, pooling_ratio)
        self.topk3 = SAGPooling(512, pooling_ratio)
        self.topk4 = SAGPooling(512, pooling_ratio)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1 + x2 + x3 + x4
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out
    
# %%

class GCN_topK(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GCNConv(dim_in, 512)
        self.gat2 = GCNConv(512*heads, 512)
        self.gat3 = GCNConv(512*heads, 512)
        self.gat4 = GCNConv(512*heads, 512)
                
        self.topk1 = TopKPooling(512, pooling_ratio)
        self.topk2 = TopKPooling(512, pooling_ratio)
        self.topk3 = TopKPooling(512, pooling_ratio)
        self.topk4 = TopKPooling(512, pooling_ratio)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1 + x2 + x3 + x4
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out
    
# %%

class GAT_model(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GATv2Conv(dim_in, 512, heads=1)
        self.gat2 = GATv2Conv(512*heads, 512, heads=1)
        self.gat3 = GATv2Conv(512*heads, 512, heads=1)
        self.gat4 = GATv2Conv(512*heads, 512, heads=1)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)


    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out
	
# %%

class GCN_model(torch.nn.Module):
    
    """Graph Attention Network for full slide graph"""
    
    def __init__(self, dim_in, heads=1, pooling_ratio=0.5):
        
        super().__init__()
        
        self.pooling_ratio = pooling_ratio
        
        self.gat1 = GCNConv(dim_in, 512)
        self.gat2 = GCNConv(512*heads, 512)
        self.gat3 = GCNConv(512*heads, 512)
        self.gat4 = GCNConv(512*heads, 512)
        
        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 2)

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index, data.batch
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out