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

import pandas as pd

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import torch.optim as optim

from torchvision import transforms

from loaders import Loaders

from training_loops import train_att_slides, train_att_multi_slide
from graph_train_loop import train_graph_slides, train_graph_multi_stain

from clam_model import VGG_embedding, GatedAttention
from Graph_model import GAT_topK

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

file = "/data/scratch/wpw030/RA/df_all_stains_patches_labels_HPC.csv"
df = pd.read_csv(file, header=0)  
df = df.dropna(subset=[label])

stains = ["CD138", "CD68", "CD20", "HE"]

# %%

file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, random_state, patient_id, label, subset)

# %%

CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST = Loaders().dictionary_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, slide_batch, num_workers)

# %%

# GRAPH
# SINGLE STAIN

sys.stdout = open("/data/home/wpw030/MangoMIL/results/Graph_single_stain_results.txt", 'w')

if train_slides:
    
    patient_stain_train = [CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN]
    patient_stain_test = [CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST]
       
    for train_stain, test_stain in zip(patient_stain_train, patient_stain_test):
        
        key = list(train_stain.values())[0].dataset.stain[0]
        classification_weights = "/data/scratch/wpw030/RA/weights/GRAPH_" + key + ".pth"
        
        embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
        graph_net = GAT_topK(1024) 
        
        if use_gpu:
            embedding_net.cuda()
            graph_net.cuda()
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(graph_net.parameters(), lr=0.0001)
        
        print(key, flush=True)

        _, graph_model = train_graph_slides(embedding_net, graph_net, train_stain, test_stain, train_ids, test_ids, loss_fn, optimizer_ft, embedding_vector_size, n_classes=n_classes, num_epochs=50)
        
        torch.save(graph_model.state_dict(), classification_weights)
        
sys.stdout.close()   

# %%


