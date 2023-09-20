# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:14:46 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np

import torch

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    
import gc 
gc.enable()


def create_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False):

    graph_dict = dict()

    embedding_net.eval()
    with torch.no_grad():

        for patient_ID, slide_loader in loader.items():
            patient_embedding = []
    
            for patch in slide_loader:
                inputs, label, _, _, _ = patch
                
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
    
                embedding = embedding_net(inputs)
                embedding = embedding.to('cpu')
                embedding = embedding.squeeze(0)
                patient_embedding.append(embedding)
    
            try:
                patient_embedding = torch.stack(patient_embedding)
            except RuntimeError:
                continue
    
            knn_graph = kneighbors_graph(patient_embedding, k, mode=mode, include_self=include_self)
            edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index)
            
            graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]
    
    return graph_dict 
