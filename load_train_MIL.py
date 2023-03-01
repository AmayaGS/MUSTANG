# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time

from collections import Counter
from collections import defaultdict

from PIL import Image
from PIL import ImageFile

import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models

from loaders import Loaders

from training_loops import train_embedding, train_att_slides, test_slides, soft_vote

from attention_models import VGG_embedding, GatedAttention
from GAT import GAT

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

subset= True

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

label = 'Pathotype'
patient_id = 'Patient ID'
n_classes=3

if n_classes > 2:
    subtyping=True
else:
    subtyping=False

# %%

file = r"C:/Users/Amaya/Documents/PhD/Data/df_all_stains_patches_labels.csv"
df = pd.read_csv(file, header=0)  
df = df.dropna(subset=[label])

stains = ["CD138", "CD68", "CD20", "HE"]

# %%

embedding_weights = r"C:/Users/Amaya/Documents/PhD/Data//embedding.pth"
classification_weights = r"C:/Users/Amaya/Documents/PhD/Data/classification.pth"

# %%

file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, random_state, patient_id, label, subset)

# %%

stain_patches_DataLoaders_TRAIN = {}
stain_patches_DataLoaders_TEST = {}
stain_patient_DataLoaders_TRAIN = {}
stain_patient_DataLoaders_TEST = {}

for stain in stains:
    
    new_key = f'{stain}'
    df_sub = df[df['Stain'] == stain]
    df_train, df_test, train_sub, test_sub = Loaders().df_loader(df_sub, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False)
    # weights for minority oversampling 
    count = Counter(df_train.labels)
    class_count = np.array(list(count.values()))
    weight = 1 / class_count
    samples_weight = np.array([weight[t] for t in df_train.labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    train_loader, test_loader = Loaders().patches_dataloader(df_train, df_test, sampler, train_batch, test_batch, num_workers, shuffle, drop_last, Loaders.collate_fn)
    train_loaded_subsets, test_loaded_subsets = Loaders().slides_dataloader(train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, label=label, patient_id=patient_id) 
       
    stain_patches_DataLoaders_TRAIN[new_key] = train_loader
    stain_patches_DataLoaders_TEST[new_key] = test_loader
    stain_patient_DataLoaders_TRAIN[new_key] = train_loaded_subsets
    stain_patient_DataLoaders_TEST[new_key] = test_loaded_subsets
    
    for outer_key, inner_values in stain_patches_DataLoaders_TRAIN.items():
        subset = {}
        subset[outer_key] = inner_values
        if outer_key == 'CD138':
            CD138_patches_TRAIN = subset.copy()
        if outer_key == 'CD68':
            CD68_patches_TRAIN = subset.copy()
        if outer_key == 'CD20':
            CD20_patches_TRAIN = subset.copy()
        if outer_key == 'HE':
            HE_patches_TRAIN = subset.copy()
            
    for outer_key, inner_values in stain_patches_DataLoaders_TEST.items():
        subset = {}
        subset[outer_key] = inner_values
        if outer_key == 'CD138':
            CD138_patches_TEST = subset.copy()
        if outer_key == 'CD68':
            CD68_patches_TEST = subset.copy()
        if outer_key == 'CD20':
            CD20_patches_TEST = subset.copy()
        if outer_key == 'HE':
            HE_patches_TEST = subset.copy()

    for outer_key, inner_values in stain_patient_DataLoaders_TRAIN.items():
        subset = {}
        for inner_key, value in inner_values.items():
            subset[inner_key] = value
        if outer_key == 'CD138':
            CD138_patients_TRAIN = subset.copy()
        if outer_key == 'CD68':
            CD68_patients_TRAIN = subset.copy()
        if outer_key == 'CD20':
            CD20_patients_TRAIN = subset.copy()
        if outer_key == 'HE':
            HE_patients_TRAIN = subset.copy()
            
    for outer_key, inner_values in stain_patient_DataLoaders_TEST.items():
        subset = {}
        for inner_key, value in inner_values.items():
            subset[inner_key] = value
        if outer_key == 'CD138':
            CD138_patients_TEST = subset.copy()
        if outer_key == 'CD68':
            CD68_patients_TEST = subset.copy()
        if outer_key == 'CD20':
            CD20_patients_TEST = subset.copy()
        if outer_key == 'HE':
            HE_patients_TEST = subset.copy()
    


# %%

if train_patches:
    
    embedding_net = models.vgg16_bn(pretrained=True)
                    
    # Freeze training for all layers
    for param in embedding_net.parameters():
        param.require_grad = False
    
    # Newly created modules have require_grad=True by default
    num_features = embedding_net.classifier[6].in_features
    features = list(embedding_net.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, embedding_vector_size)])
    features.extend([nn.Dropout(0.5)])
    features.extend([nn.Linear(embedding_vector_size, n_classes)]) # Add our layer with n outputs
    embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier

    if use_gpu:
        embedding_net.cuda() 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(embedding_net.parameters(), lr=0.0001, momentum=0.9)

# %%

if train_patches:
    
    patches_stain_train = [CD138_patches_TRAIN, CD68_patches_TRAIN, CD20_patches_TRAIN, HE_patches_TRAIN]
    patches_stain_test = [CD138_patches_TEST, CD68_patches_TEST, CD20_patches_TEST, HE_patches_TEST]

    for train_stain, test_stain in zip(patches_stain_train, patches_stain_test):
        key, _ = list(train_stain.items())[0]
        embedding_weights = r"C:/Users/Amaya/Documents/PhD/Data//embedding_patches_" + key + ".pth"
        model = train_embedding(embedding_net, train_stain, test_stain, criterion, optimizer, num_epochs=1)
        torch.save(model.state_dict(), embedding_weights)

# %%

if train_slides:

    # embedding_weights_CD138 = r"C:/Users/Amaya/Documents/PhD/Data//embedding_patches_" + 'CD138' + ".pth"
    # embedding_net_CD138 = VGG_embedding(embedding_weights_CD138, finetuned=True, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    # #CD68
    # embedding_weights_CD68 = r"C:/Users/Amaya/Documents/PhD/Data//embedding_patches_" + 'CD68' + ".pth"
    # embedding_net_CD68 = VGG_embedding(embedding_weights_CD68, finetuned=True, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    # #CD20
    # embedding_weights_CD20 = r"C:/Users/Amaya/Documents/PhD/Data//embedding_patches_" + 'CD20' + ".pth"
    # embedding_net_CD20 = VGG_embedding(embedding_weights_CD20, finetuned=True, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    # #HE 
    # embedding_weights_HE = r"C:/Users/Amaya/Documents/PhD/Data//embedding_patches_" + 'HE' + ".pth"
    # embedding_net_HE = VGG_embedding(embedding_weights_HE, finetuned=True, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    
    
    # classification_net_CD138 = GatedAttention(n_classes=n_classes, subtyping=subtyping)
    # #CD68
    # classification_net_CD68 = GatedAttention(n_classes=n_classes, subtyping=subtyping)
    # #CD20
    # classification_net_CD20 = GatedAttention(n_classes=n_classes, subtyping=subtyping)
    # #HE 
    # classification_net_HE = GatedAttention(n_classes=n_classes, subtyping=subtyping)
   
    embedding_net = VGG_embedding(embedding_weights, finetuned=False, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    classification_net = GatedAttention(n_classes=n_classes, subtyping=subtyping) # add classification weight variable. 
    
    if use_gpu:
        embedding_net.cuda()
        classification_net.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(classification_net.parameters(), lr=0.0001)

# %%
    
if train_slides:
    
    embedding_model, classification_model = train_att_slides(embedding_net, classification_net, CD138_patients_TRAIN, CD138_patients_TEST, train_ids, test_ids, loss_fn, optimizer_ft, embedding_vector_size, n_classes=n_classes, bag_weight=0.7, num_epochs=10)
    torch.save(classification_model.state_dict(), classification_weights)
    
# %%

stain_loaders = [CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN]

# %%
    
###################################
# TRAIN

embedding_net.eval()
classification_net.train(True)

list_patient_multi_stain = []
labels = []
all_labels = []

for batch_idx, loader in enumerate(zip(CD138_patients_TRAIN.values(), CD68_patients_TRAIN.values(), CD20_patients_TRAIN.values(), HE_patients_TRAIN.values())):
    
    print()
    print("\rPatient {}/{}".format(batch_idx, len(train_ids)), end='', flush=True)
    print()
    optimizer_ft.zero_grad()
    
    patient_embedding = []
    labels = []
    
    for i, data in enumerate(loader):
        
        print("\rStain {}/{}".format(i, len(loader)), end='', flush=True)
        
        slide_embedding = []
        
        for patch in data:
            
            inputs, label = patch
    
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
            #print(len(patient_embedding))
            slide_embedding = slide_embedding.cuda()
                    
            logits, Y_prob, Y_hat, _, instance_dict, feature_vector = classification_net(slide_embedding, label=label, instance_eval=True)
            patient_embedding.append(feature_vector[0].detach().to('cpu'))
                    
        except RuntimeError:
            
            patient_embedding.append(torch.zeros(embedding_vector_size))
         
        labels.append(label.type(torch.LongTensor))                 
    
    all_labels.append(labels)
    patient_multi_stain = torch.stack(patient_embedding)
    list_patient_multi_stain.append(patient_multi_stain)

#logits, Y_prob, Y_hat, _, instance_dict, feature_vector = classification_net(patient_embedding, label=label, instance_eval=True)

#%%

net = GAT(1024, 64, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(net.parameters(), lr=0.0001)

# %%

graph_object = []

for i, patient in enumerate(zip(list_patient_multi_stain, all_labels)):
    
    node_features = list_patient_multi_stain[i]
    num_nodes = node_features.size(0)
    labels = all_labels[i]
    edge_index = torch.zeros((2, num_nodes * (num_nodes - 1)))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index[0, i * (num_nodes - 1) + j - (1 if j > i else 0)] = i
                edge_index[1, i * (num_nodes - 1) + j - (1 if j > i else 0)] = j
    graph = Data(x=node_features, edge_index=edge_index.type(torch.LongTensor), y=labels)
    graph_object.append(graph)
    
# %%

h, f = net(graph)

# %%

from torch_geometric.utils import to_networkx
G = to_networkx(graph, to_undirected=True)
visualize_graph(G, color=graph.y)
visualize_embedding(h, graph.y)

# %%


correct = 0
    
for graph in graph_object:
    
    out, h = net(graph.x, graph.edge_index)
    pred = out.argmax(dim=1)
    loss = loss_fn(out, torch.zeros(4).to(torch.long))  # Compute the loss solely based on the training nodes.
    correct += int((pred == torch.stack(graph.y).to(torch.long).transpose(0, 1)).sum())
    loss.backward()  # Derive gradients.
    optimizer_ft.step()
    
acc =  correct / (len(graph_object) * 4 )



# %%

# if train_slides:
    
#     embedding_model, classification_model = train_att_slides(embedding_net, classification_net, train_ids, test_ids,  CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST, loss_fn, optimizer_ft, embedding_vector_size, n_classes=n_classes, bag_weight=0.7, num_epochs=10)
#     torch.save(classification_model.state_dict(), classification_weights)


# %%

if testing_slides:
    
    loss_fn = nn.CrossEntropyLoss()
    
    embedding_net = VGG_embedding(embedding_weights, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    classification_net = GatedAttention(n_classes=n_classes, subtyping=subtyping)

    classification_net.load_state_dict(torch.load(classification_weights), strict=True)
    
    if use_gpu:
        embedding_net.cuda()
        classification_net.cuda()

# %%

if testing_slides:
    
    test_error, test_auc, test_accuracy, test_acc_logger, labels, prob, clsf_report, conf_matrix, sensitivity, specificity, incorrect_preds =       test_slides(embedding_net, classification_net, test_loaded_subsets, loss_fn, n_classes=2)

# %%

target_names=["Fibroid", "M/Lymphoid"]

auc_plot(labels, prob[:, 1], test_auc)
pr_plot(labels, prob[:, 1], sensitivity, specificity)
plot_confusion_matrix(conf_matrix, target_names, title='Confusion matrix', cmap=None, normalize=True)


###############################
# %%

history = soft_vote(embedding_net, test_loaded_subsets)

# %%

