# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:34:24 2023

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

from auxiliary_functions import Accuracy_Logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    
import gc 
gc.enable()

# %%


def train_graph_multi_stain(embedding_net, graph_net, train_loader, test_loader, loss_fn, optimizer, K, embedding_vector_size, n_classes, num_epochs=1, training=True, testing=True, random_seed=str(2), heads=str(1), pooling_ratio=str(0.5), learning_rate=str(0.0001), checkpoints="PATH_checkpoints"):
    
    since = time.time()
    best_acc = 0.
    best_AUC = 0.
    
    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []
    
    for epoch in range(num_epochs):
        
        ##################################
        # TRAIN
        
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        train_acc = 0 
        train_count = 0
    
        if training:
        
            print("Epoch {}/{}".format(epoch, num_epochs), flush=True)
            print('-' * 10)
    
            embedding_net.eval()
            graph_net.train(True)
            
            for batch_idx, graph_loader in train_loader:
                
                data, label = graph_loader
                
                if use_gpu:
                    data, label = data.cuda(), label.cuda()
                else:
                    data, label = data, label
                
                
                logits, Y_prob = graph_net(data)
                Y_hat = Y_prob.argmax(dim=1)
                acc_logger.log(Y_hat, label)
                loss = loss_fn(logits, label)
                
                train_acc += torch.sum(Y_hat == label.data)
                train_count += 1

                if (batch_idx + 1) % 20 == 0:
                    print('- batch {}, loss: {:.4f}, '.format(batch_idx, loss) + 
                        'label: {}, bag_size: {}'.format(label.item(), data.size(0))) # CHECK HERE
                
                # backward pass
                loss.backward()
                # step
                optimizer.step()
                optimizer.zero_grad()
                
                del data, knn_graph, edge_index, slide_embedding, patient_embedding, logits, Y_prob, Y_hat
                gc.collect()
                
            total_loss = loss.item() / train_count
            train_accuracy =  train_acc / train_count
            
            print()
            print('Epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(epoch, total_loss, train_accuracy))
            for i in range(n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)
                      
            graph_net.train(False)
        
        ################################
        # TEST
        
        if testing:
        
            embedding_net.eval()
            graph_net.eval()
            
            val_acc_logger = Accuracy_Logger(n_classes)
            val_loss = 0.
            val_acc = 0
            val_count = 0 
            
            prob = []
            labels = []
    
            for batch_idx, loader in enumerate(zip(CD138_patients_TEST.values(), CD68_patients_TEST.values(), CD20_patients_TEST.values(), HE_patients_TEST.values())):
    
                patient_embedding = []
                
                for i, data in enumerate(loader):
                
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
                        patient_embedding.append(slide_embedding)
                    
                    except RuntimeError:
                        continue            
                try:
                    
                    patient_embedding = torch.cat(patient_embedding)
                    
                except RuntimeError:
                    continue
                
                knn_graph = kneighbors_graph(patient_embedding, K, mode='connectivity', include_self=False)
                edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
                data = Data(x=patient_embedding, edge_index=edge_index)
                
                logits, Y_prob = graph_net(data.cuda())
                Y_hat = Y_prob.argmax(dim=1)
                val_acc_logger.log(Y_hat, label)
                
                val_acc += torch.sum(Y_hat == label.data)
                val_count += 1
                
                loss = loss_fn(logits, label)
                val_loss += loss.item()
                                
                prob.append(Y_prob.detach().to('cpu').numpy())
                labels.append(label.item())
                
                del data, knn_graph, edge_index, slide_embedding, patient_embedding, logits, Y_prob, Y_hat
                gc.collect()
                       
            val_loss /= val_count
            val_accuracy = val_acc / val_count
            
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy.item())
                   
            if n_classes == 2:
                prob =  np.stack(prob, axis=1)[0]
                val_auc = roc_auc_score(labels, prob[:, 1])
                aucs = []
            else:
                aucs = []
                binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
                prob =  np.stack(prob, axis=1)[0]
                for class_idx in range(n_classes):
                    if class_idx in labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                        aucs.append(calc_auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
            
                val_auc = np.nanmean(np.array(aucs))
            
            val_auc_list.append(val_auc)
            
            clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
            conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
                        
            print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_auc, val_accuracy), flush=True)
        
            for i in range(n_classes):
                acc, correct, count = val_acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                
            print(clsf_report)
            print(conf_matrix)
            
            if n_classes == 2:
                sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
                specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1]) 
                print('Sensitivity: ', sensitivity) 
                print('Specificity: ', specificity) 
    
            if val_accuracy >= best_acc:
                if val_auc >= best_AUC:
                    best_acc = val_accuracy
                    best_AUC = val_auc
                    checkpoint_weights = checkpoints + random_seed + "_" + heads + "_" + pooling_ratio + "_" + learning_rate + "_checkpoint_" + str(epoch) + ".pth"    
                    torch.save(graph_net.state_dict(), checkpoint_weights)
                       
    elapsed_time = time.time() - since
      
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    graph_net.load_state_dict(torch.load(checkpoint_weights), strict=True)
        
    return val_loss_list, val_accuracy_list, val_auc_list, graph_net
    

# TEST

def test_multi_stain_wsi(embedding_net, graph_net, CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST, train_ids, test_ids, loss_fn, optimizer, K, n_classes=2):

    since = time.time()

    embedding_net.eval()
    graph_net.eval()

    test_count = 0 

    val_acc_logger = Accuracy_Logger(n_classes)
    val_loss = 0.
    val_acc = 0

    prob = []
    labels = []

    for batch_idx, loader in enumerate(zip(CD138_patients_TEST.values(), CD68_patients_TEST.values(), CD20_patients_TEST.values(), HE_patients_TEST.values())):

        patient_embedding = []

        for i, data in enumerate(loader):

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
                patient_embedding.append(slide_embedding)
            
            except RuntimeError:
                continue
        try:
            
            patient_embedding = torch.cat(patient_embedding)
            
        except RuntimeError:
            continue
        
        knn_graph = kneighbors_graph(patient_embedding, K, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
        data = Data(x=patient_embedding, edge_index=edge_index)
        
        logits, Y_prob = graph_net(data.cuda())
        Y_hat = Y_prob.argmax(dim=1)
        val_acc_logger.log(Y_hat, label)
        
        val_acc += torch.sum(Y_hat == label.data)
        test_count += 1
        
        loss = loss_fn(logits, label)
        val_loss += loss.item()
                        
        #Y_prob = torch.exp(Y_prob)
        prob.append(Y_prob.detach().to('cpu').numpy())
        labels.append(label.item())
        
        del data, knn_graph, edge_index, slide_embedding, patient_embedding, logits, Y_prob, Y_hat
        gc.collect()
               
    val_loss /= test_count
    val_accuracy = val_acc / test_count
           
    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
        val_auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        prob =  np.stack(prob, axis=1)[0]
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
    
        val_auc = np.nanmean(np.array(aucs))
    
    clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
                
    print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_auc, val_accuracy), flush=True)
    
    for i in range(n_classes):
        acc, correct, count = val_acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
    print(clsf_report)
    print(conf_matrix)
    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1]) 
        print('Sensitivity: ', sensitivity) 
        print('Specificity: ', specificity) 
      
    elapsed_time = time.time() - since
      
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return val_auc, val_accuracy, labels, prob, clsf_report, conf_matrix, sensitivity, specificity