# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot as plt

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from loaders import Loaders
from create_store_graphs import create_graphs
from graph_train_loop import train_graph_multi_stain, test_multi_stain_wsi

from Graph_model import VGG_embedding, GAT_SAGPool

#from plotting_results import auc_plot, pr_plot, plot_confusion_matrix

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

PATH_patches =  r"C:\Users\Amaya\Documents\PhD\Data\df_all_stains_patches_labels.csv"  # csv with file location is foud here 
PATH_output_file = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\results\GRAPH_multi_seed_" # keep output results here
PATH_output_weights = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\weights"  # keep output weights here 
PATH_checkpoints = r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights" # keep checkpoints here 

# %%

# image transforms 

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

# parameters 

state = 42
torch.manual_seed(state)
train_fraction = .7

subset= False
slide_batch = 1

K=5

num_workers = 0

creating_knng = False
training = True
testing = True

embedding_vector_size = 1024
learning_rate = 0.0001

str_lr = str(learning_rate)
str_state = str(state)

pooling_ratio = 0.7
heads = 2

str_pr = str(pooling_ratio)
str_hd = str(heads)

label = 'Pathotype_binary'
patient_id = 'Patient ID'
n_classes=2


# %%

df = pd.read_csv(PATH_patches, header=0)
df = df.dropna(subset=[label])
    
# %%

# create k-NNG with VGG patch embedddings 
if creating_knng:

    file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, state, patient_id, label, subset)

    df_train, df_test, train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False)

    train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=True, label='Pathotype_binary', patient_id="Patient ID")

    embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    if use_gpu:
        embedding_net.cuda()

    train_graph_dict = create_graphs(embedding_net, train_slides, k=5, mode='connectivity', include_self=False)
    test_graph_dict = create_graphs(embedding_net, test_slides, k=5, mode='connectivity', include_self=False)

    # save k-NNG with VGG patch embedddings for future use
    print("Started saving train_graph_dict to file")
    with open("train_graph_dict.pkl", "wb") as file:
        pickle.dump(train_graph_dict, file)  # encode dict into JSON
        print("Done writing dict into pickle file")
        
    print("Started saving test_graph_dict to file")
    with open("test_graph_dict.pkl", "wb") as file:
        pickle.dump(test_graph_dict, file)  
        print("Done writing dict into .pickle file")


if creating_knng==False:

    with open("train_graph_dict.pkl", "rb") as train_file:
    # Load the dictionary from the file
        train_graph_dict = pickle.load(train_file)

    with open("test_graph_dict.pkl", "rb") as test_file:
    # Load the dictionary from the file
        test_graph_dict = pickle.load(test_file)

# %%

# MULTI-STAIN GRAPH

#sys.stdout = open(PATH_output_file + str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".txt", 'a')

#classification_weights = PATH_output_weights + "\best" + str_hd + ".pth"

graph_net = GAT_SAGPool(embedding_vector_size, heads=heads, pooling_ratio=pooling_ratio)
loss_fn = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(graph_net.parameters(), lr=learning_rate)

#print(state, heads, pooling_ratio, learning_rate, flush=True)

if use_gpu:
     graph_net.cuda()

val_loss, val_accuracy, val_auc, graph_weights = train_graph_multi_stain(graph_net, train_graph_dict, test_graph_dict, loss_fn, optimizer_ft, K, embedding_vector_size, n_classes, num_epochs=50, training=training, testing=testing, random_seed=str_state, heads=str_hd, pooling_ratio=str_pr, learning_rate=str_lr, checkpoint=False, checkpoints=PATH_checkpoints)

# torch.save(graph_weights.state_dict(), classification_weights)

#print(test_loss)
#print(test_accuracy)

np.savetxt(r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights\training results\test_loss_graph_" +  str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", val_loss)
np.savetxt(r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights\training results\test_accuracy_graph_" +  str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", val_accuracy)

#sys.stdout.close()

