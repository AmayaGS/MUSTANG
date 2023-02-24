# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:54:31 2023

@author: AmayaGS
"""

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
import networkx as nx

#%%

k = 2  # Number of nearest neighbors to connect

# Generate data
# Generate data
num_patches = 10
patch_features = torch.randn(num_patches, 256)  # Feature matrix of patches

# Compute k-nearest neighbors graph
knn_graph = kneighbors_graph(patch_features, k, mode='connectivity', include_self=False)
edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(x=patch_features, edge_index=edge_index.t().contiguous())

# Print data object
# print(data)
# patch_features
# patch_features[0]
# patch_features[0].shape

# Compute k-nearest neighbors graph
knn_graph = kneighbors_graph(patch_features, k, mode='connectivity', include_self=False)
edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(x=patch_features, edge_index=edge_index.t().contiguous())

# Convert to NetworkX graph
G = nx.Graph()
G.add_nodes_from(range(patch_features.shape[0]))
G.add_edges_from(edge_index.t().tolist())

# Plot the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

