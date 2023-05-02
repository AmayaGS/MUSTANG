# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:04:40 2023

@author: AmayaGS
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

import time
import os
import copy
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from auxiliary_functions import Accuracy_Logger
from clam_model import VGG_embedding, GatedAttention
from Graph_model import GAT_TopK
from loaders import Loaders
from graph_train_loop import train_graph_multi_stain
from plotting_results import auc_plot, pr_plot, plot_confusion_matrix

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()

import gc 
gc.enable()

# %%

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        #transforms.ColorJitter(brightness=0.005, contrast=0.005, saturation=0.005, hue=0.005),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1), 
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.1)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

# %%

torch.manual_seed(42)
train_fraction = .7
random_state = 2

subset= False

train_batch = 10
test_batch = 1
slide_batch = 1

num_workers = 0
shuffle = False
drop_last = False

train_patches = False
train_slides = True
testing_slides = True

finetuned = False 
embedding_vector_size = 1024

#subtyping = False # (True for 3 class problem) 

# %%

label = 'Pathotype_binary'
patient_id = 'Patient ID'
n_classes=2

if n_classes > 2:
    subtyping=True
else:
    subtyping=False

# %%

file = r"C:\Users\Amaya\Documents\PhD\Data\df_all_stains_patches_labels.csv"
df = pd.read_csv(file, header=0)  
df = df.dropna(subset=[label])

stains = ["CD138", "CD68", "CD20", "HE"]

# %%

file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, random_state, patient_id, label, subset)

# %%

CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST = Loaders().dictionary_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, slide_batch, num_workers)

#%%

embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
embedding_net.cuda()

#%%

loader = zip(CD138_patients_TRAIN.values(), CD68_patients_TRAIN.values(), CD20_patients_TRAIN.values(), HE_patients_TRAIN.values())

#%%

data = next(zip(loader))

# %%
embedding_net.eval()
        
patient_embedding = [] 
patches = []

for i, slides in enumerate(data):
                
# 	stain = slides.dataset.stain[0]
# 	patient = slides.dataset.filepaths[0].split("\\")[-1].split("_")[0]
                
# 	print("\rStain {}/{} - {} - {}".format(i, len(data), stain, patient), end='', flush=True)

	for slide in slides:
		
		slide_embedding = []
		
		for patch in slide:
		
			inputs, label = patch
			
			patches.append([inputs, slide.dataset.stain[0]])
	            
			if use_gpu:
				inputs, label = inputs.cuda(), label.cuda()
			else:
				inputs, label = inputs, label
	            
			embedding = embedding_net(inputs)
	            
			embedding = embedding.detach().to('cpu')
			embedding = embedding.squeeze(0)
			slide_embedding.append(embedding)
                    
		try:
	                
			slide_embedding = torch.stack(slide_embedding)
			patient_embedding.append(slide_embedding)
	                
		except RuntimeError:
			continue
                
# try:
# 	patient_embedding = torch.cat(patient_embedding)            
# except RuntimeError: 
#     continue
patient_embedding = torch.cat(patient_embedding) 

# %%    
       
knn_graph = kneighbors_graph(patient_embedding, 5, mode='connectivity', include_self=False)
edge_index = torch.tensor(np.array(knn_graph.nonzero()),dtype=torch.long)

# %%

graph_model = Data(x=patient_embedding, edge_index=edge_index)
		
#%%

from networkx.drawing.nx_agraph import graphviz_layout

# %%

graph = to_networkx(graph_model)

# Plot the graph
pos = nx.nx_agraph.graphviz_layout(graph, prog="fdp")

#pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=False, node_size=1, width=0.1, alpha=0.4, arrows=False)
plt.show()


# %%

# loader_colors = {'CD138': 'orange', 'CD68': 'red', 'CD20': 'blue', 'HE': 'pink'}

# graph = to_networkx(data)

# # compute a force-directed layout for the graph
# layout = nx.spring_layout(graph, weight=None)

# # create the scatter plot with icons
# fig, ax = plt.subplots(figsize=(50, 50))
# for i, feature in enumerate(patient_embedding):
#     x, y = layout[i]
#     patch_i = patches[i][0]
#     stain_i = patches[i][1]
#     color = loader_colors[stain_i]
#     #if stain_i == 'CD138':
#     patch_ar = np.asarray(patch_i.squeeze(0).permute(1, 2, 0)) 
#     icon = Image.fromarray((patch_ar * 255).astype(np.uint8)).resize((30, 30))
#     rect = plt.Rectangle((x-0.015, y-0.015), 0.03, 0.03, linewidth=1, edgecolor=color, facecolor='none')
#     ax.add_patch(rect)
#     ax.imshow(icon, extent=[x-0.015, x+0.015, y-0.015, y+0.015], alpha=0.8)

# X = [layout[node][0] for node in layout.keys()]
# Y = [layout[node][1] for node in layout.keys()]

# ax.scatter(X, Y, alpha=0.7, s=50)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

# %%
  
# knn_graph = kneighbors_graph(patient_embedding, 5, mode='connectivity', include_self=False, metric="minkowski")
# edge_index = torch.tensor(np.array(knn_graph.nonzero()),dtype=torch.long)
# data = Data(x=patient_embedding, edge_index=edge_index)

#%%

loader_colors = {'CD138': 'orange', 'CD68': 'red', 'CD20': 'blue', 'HE': 'purple'}

graph = to_networkx(graph_model)

# Plot the graph
layout = nx.spring_layout(graph)

# create the scatter plot with icons
fig, ax = plt.subplots(figsize=(50, 50))
node_colors = [loader_colors[patches[i][1]] for i in range(len(patches))]

X = [layout[node][0] for node in layout.keys()]
Y = [layout[node][1] for node in layout.keys()]

ax.scatter(X, Y, alpha=1, s=250, c=node_colors)

# add a legend for the loader colors
handles = [plt.Rectangle((0,0),1,1, color=loader_colors[key]) for key in loader_colors]
labels = loader_colors.keys()
ax.legend(handles, labels, loc='upper right', fontsize=70)

edge_colors = ['gray' for u, v in graph.edges()]

nx.draw_networkx_edges(graph, pos=layout, edge_color= edge_colors, alpha=0.5, arrows=False)

plt.axis('off')
plt.show()


# %%

graph = to_networkx(data)

# compute a force-directed layout for the graph
layout = nx.spring_layout(graph, weight=None)

# create the scatter plot with icons
fig, ax = plt.subplots(figsize=(20, 20))
for i, feature in enumerate(patient_embedding):
    x, y = layout[i]
    patch_i = patches[i][0]
    stain_i = patches[i][1]
    color = loader_colors[stain_i]
    patch_ar = np.asarray(patch_i.squeeze(0).permute(1, 2, 0)) 
    icon = Image.fromarray((patch_ar * 255).astype(np.uint8)).resize((40, 40))
    rect = plt.Rectangle((x-0.015, y-0.015), 0.03, 0.03, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.imshow(icon, extent=[x-0.015, x+0.015, y-0.015, y+0.015], alpha=0.8, resample=0.5)

X = [layout[node][0] for node in layout.keys()]
Y = [layout[node][1] for node in layout.keys()]

ax.scatter(X, Y, alpha=0.7, s=1, edgecolors='none', linewidths=1, facecolors='none')
#ax.set_xlabel('X', fontsize=20)
#ax.set_ylabel('Y', fontsize=20)
#ax.tick_params(axis='both', labelsize=16)
#ax.set_xlim([-0.6,0.6])
#ax.set_ylim([-0.6,0.6])
plt.tight_layout()
plt.show()

#his code adds a thicker border to the coloured rectangles, removes the transparency of the patch icons, increases the size of the plot and axes labels, adds edgecolors and linewidths to the scatter plot markers, sets the axis tick labels to a larger size, and sets the plot limits to the range [-0.6, 0.6] on both axes. Finally, plt.tight_layout() is called to ensure that all elements of the plot fit within the figure.







