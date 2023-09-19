# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys

from PIL import Image
from PIL import ImageFile

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import torch.optim as optim

from torchvision import transforms

from loaders import Loaders

#from training_loops import train_att_slides, train_att_multi_slide
from graph_train_loop import train_graph_slides, train_graph_multi_stain, test_multi_stain_wsi

from clam_model import VGG_embedding, GatedAttention
from Graph_model import GAT_SAGPool, GAT_SAGPool_mha, GAT_SAGPool_pooling

from plotting_results import auc_plot, pr_plot, plot_confusion_matrix

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()  

import gc 
gc.enable()

# %%

PATH_patches =  r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\df_all_stains_patches_labels_HPC.csv"
PATH_output_file = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\results\GRAPH_multi_seed_"
PATH_output_weights = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\weights" 
PATH_checkpoints = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\weights\checkpoints"

# %%

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
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


#random_state = [42, 8, 33, 107, 1, 58, 11, 14, 6, 9]
random_state = [2]

for state in random_state:
    
    torch.manual_seed(state)
    train_fraction = .7
    random = state
    
    subset= False
    slide_batch = 1
    
    K=5
    
    num_workers = 0

    training = True
    testing = True
    
    embedding_vector_size = 1024
    learning_rate = 0.0001
    
    str_lr = str(learning_rate)
    str_random = str(random)

    pooling_ratio = 0.5
    heads = 2

    str_pr = str(pooling_ratio)
    str_hd = str(heads)

    label = 'Pathotype_binary'
    patient_id = 'Patient ID'
    n_classes=2
    
    if n_classes > 2:
        subtyping=True
    else:
        subtyping=False
    
    df = pd.read_csv(PATH_patches, header=0)  
    df = df.dropna(subset=[label])
    
    stains = ["CD138", "CD68", "CD20", "HE"]
    
    file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, random, patient_id, label, subset)
    

    CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST = Loaders().dictionary_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, slide_batch, num_workers)
  
  # GRAPH
  # MULTI STAIN
  
    sys.stdout = open(PATH_output_file + str_random + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".txt", 'a')
               
    classification_weights = PATH_output_weights + "\best str_hd + ".pth"
    
    embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    graph_net = GAT_SAGPool(1024, heads=heads, pooling_ratio=pooling_ratio) 
    
    print(state, random, heads, pooling_ratio, learning_rate, flush=True)
    
    if use_gpu:
        embedding_net.cuda()
        graph_net.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(graph_net.parameters(), lr=learning_rate)

    val_loss, val_accuracy, val_auc, graph_weights = train_graph_multi_stain(embedding_net, graph_net, CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST, train_ids, test_ids, loss_fn, optimizer_ft, K, embedding_vector_size, n_classes, num_epochs=50, training=training, testing=testing, random_seed=str_random, heads=str_hd, pooling_ratio=str_pr, learning_rate=str_lr, checkpoints=PATH_checkpoints)
    
    torch.save(graph_weights.state_dict(), classification_weights)
    
    print(test_loss)
    print(test_accuracy)
    
    np.savetxt("/data/home/wpw030/MangoMIL/results/test_loss_graph_" +  str_random + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", test_loss)
    np.savetxt("/data/home/wpw030/MangoMIL/results/test_accuracy_graph_" +  str_random + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", test_accuracy)
            
    sys.stdout.close()        
        
  
