# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import pickle

from PIL import Image
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from loaders import Loaders
from embedding_net import VGG_embedding
from create_store_graphs import create_embeddings_graphs
from graph_train_loop import train_graph_multi_wsi
from auxiliary_functions import seed_everything
from Graph_model import GAT_SAGPool

from plotting_results import auc_plot, pr_plot, confusion_matrix_plot, learning_curve_plot

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

# set working directory
os.chdir(r"C:\Users\Amaya\Documents\PhD\MangoMIL")

# %%

# dataset_name = "Sjogren"
# PATH_patches = r"C:\Users\Amaya\Documents\PhD\NECCESITY\Slides\qj_patch_labels.csv"
#PATH_checkpoints = r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights"
#file = r"C:/Users/Amaya/Documents/PhD/Data/" + stain + "/df_all_"+ stain + "_patches_labels.csv"
#df = pd.read_csv(PATH_patches, header=0)

 # %%

dataset_name = "RA"
PATH_patches =  r"C:\Users\Amaya\Documents\PhD\Data\df_all_stains_patches_labels.csv"  # csv with file location is foud here
PATH_output_file = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\results\GRAPH_multi_seed_" # keep output results here
PATH_output_weights = r"C:\Users\AmayaGS\Documents\PhD\MangoMIL\weights"  # keep output weights here
PATH_checkpoints = r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights" # keep checkpoints here

# %%

# image transforms
train_transform = transforms.Compose([
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
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# %%

# parameters

state = 42
seed_everything(state)
train_fraction = .7

subset= False
slide_batch = 10 # this needs to be larger than one, otherwise Dataloader can fail when only passed a None object from collate function. Should change Dataset to Iterable dataset instead to solve this problem.

K=5

num_workers = 0

creating_knng = False
creating_embedding = False
training = True
testing = True

train_graph = True
train_clam = True

embedding_vector_size = 1024
learning_rate = 0.0001

str_lr = str(learning_rate)
str_state = str(state)

pooling_ratio = 0.7
heads = 2

str_pr = str(pooling_ratio)
str_hd = str(heads)

label = 'Pathotype_binary'
#label = "Binary disease"
patient_id = 'Patient ID'
n_classes=2

graph_parameters = PATH_checkpoints + "\\graph_" + dataset_name + "_" + str_state + "_" + str_hd + "_" + str_pr + "_" + str_lr

clam_parameters = PATH_checkpoints + "\\clam_"  + dataset_name + "_" + str_state + "_" + str_hd + "_" + str_pr + "_" + str_lr


# %%

df = pd.read_csv(PATH_patches, header=0)
df = df.dropna(subset=[label])

# %%

def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# %%

# create k-NNG with VGG patch embedddings

if creating_knng:

    file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, state, patient_id, label, subset)

    train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=subset)

    train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)

    embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    if use_gpu:
        embedding_net.cuda()

    # save k-NNG with VGG patch embedddings for future use
    slides_dict = {('train_graph_dict_', 'train_embedding_dict_') : train_slides ,
                   ('test_graph_dict_', 'test_embedding_dict_'): test_slides}

    for file_prefix, slides in slides_dict.items():

        graph_dict, embedding_dict = create_embeddings_graphs(embedding_net, slides, k=5, mode='connectivity', include_self=False)

        print("Started saving %s to file" % file_prefix[0])
        with open(file_prefix[0] + dataset_name + ".pkl", "wb") as file:
            pickle.dump(graph_dict, file)  # encode dict into Pickle
            print("Done writing graph dict into pickle file")

        print("Started saving %s to file" % file_prefix[1])
        with open(file_prefix[1] + dataset_name + ".pkl", "wb") as file:
            pickle.dump(embedding_dict, file)  # encode dict into Pickle
            print("Done writing embedding dict into pickle file")

# %%

# load pickled embeddings and graphs

if not creating_knng:

    with open("train_graph_dict_" + dataset_name + ".pkl", "rb") as train_file:
    # Load the dictionary from the file
        train_graph_dict = pickle.load(train_file)

    with open("test_graph_dict_" + dataset_name + ".pkl", "rb") as test_file:
    # Load the dictionary from the file
        test_graph_dict = pickle.load(test_file)


if not creating_embedding:

    with open("train_embedding_dict_" + dataset_name + ".pkl", "rb") as train_file:
    # Load the dictionary from the file
        train_embedding_dict = pickle.load(train_file)

    with open("test_embedding_dict_" + dataset_name + ".pkl", "rb") as test_file:
    # Load the dictionary from the file
        test_embedding_dict = pickle.load(test_file)

# %%

# calculate weights for minority oversampling

count = []
for k, v in train_graph_dict.items():
    count.append(v[1].item())
counter = Counter(count)
class_count = np.array(list(counter.values()))
weight = 1 / class_count
samples_weight = np.array([weight[t] for t in count])
samples_weight = torch.from_numpy(samples_weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples=len(samples_weight),  replacement=True)

# %%

# MULTI-STAIN GRAPH

#sys.stdout = open(PATH_output_file + str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".txt", 'a')

#classification_weights = PATH_output_weights + "\best" + str_hd + ".pth"

train_graph_loader = torch.utils.data.DataLoader(train_graph_dict, batch_size=1, shuffle=False, num_workers=0, sampler=sampler, drop_last=False)
#train_graph_loader = torch_geometric.loader.DataLoader(train_graph_dict, batch_size=1, shuffle=False, num_workers=0, sampler=sampler, drop_last=False, generator=seed_everything(state))
test_graph_loader = torch.utils.data.DataLoader(test_graph_dict, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

if train_graph:
    graph_net = GAT_SAGPool(embedding_vector_size, heads=heads, pooling_ratio=pooling_ratio)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(graph_net.parameters(), lr=learning_rate)
    if use_gpu:
        graph_net.cuda()

    #print(state, heads, pooling_ratio, learning_rate, flush=True)

    graph_weights, results_dict = train_graph_multi_wsi(graph_net, train_graph_loader, test_graph_loader, loss_fn, optimizer_ft, n_classes, num_epochs=20, training=training, testing=testing, checkpoint=False, checkpoint_path= graph_parameters + "_checkpoint_")

    #torch.save(graph_weights.state_dict(), classification_weights)

    df_results = pd.DataFrame.from_dict(results_dict)
    df_results.to_csv(graph_parameters + ".csv", index=False)
    learning_curve_plot(df_results)

    #sys.stdout.close()

# if train_clam:
#     clam_net = GatedAttention(embedding_vector_size)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer_ft = optim.Adam(clam_net.parameters(), lr=learning_rate)
#     if use_gpu:
#         clam_net.cuda()


#     val_loss_list, val_accuracy_list, val_auc_list, clam_weights = train_clam_multi_slide(clam_net, train_embedding_dict, test_embedding_dict, loss_fn, optimizer_ft, embedding_vector_size, n_classes, bag_weight=0.7, num_epochs=50, training=training, testing=testing, checkpoint=False, checkpoint_path=clam_parameters + "_checkpoint_")

#     np.savetxt(r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights\training results\test_loss_clam_" + dataset_name + "_" + str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", val_loss)
#     np.savetxt(r"C:\Users\Amaya\Documents\PhD\MangoMIL\weights\training results\test_accuracy_clam_" + dataset_name + "_" +  str_state + "_heads_" + str_hd + "_" + str_pr + "_" + str_lr + ".csv", val_accuracy)