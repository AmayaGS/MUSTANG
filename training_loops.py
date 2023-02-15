# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:58:10 2023

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import copy
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch
import torch.nn.functional as F

#from torch.optim import lr_scheduler
#from torch.autograd import Variable

from auxiliary_functions import Accuracy_Logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    
import gc 
gc.enable()

    

def train_embedding(vgg, train_loader, test_loader, criterion, optimizer, num_epochs=1):
    
    since = time.time()
    #best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_loader)
    val_batches = len(test_loader)
        
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        total_train = 0
        total_test= 0
        
        # TRAINING
        vgg.train(True)

        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            total_train += 1
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train  / len(train_loader) # wrong size
        avg_acc = acc_train / len(train_loader)
        
        vgg.train(False)
        
        #TESTING
        vgg.eval()

        for i, data in enumerate(test_loader):
            if i % 10 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
                
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)    
            total_test += 1
                        
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(test_loader)
        avg_acc_val = acc_val / len(test_loader)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
            
    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Lowest loss: {:.2f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    
    return vgg


def train_att_slides(embedding_net, classification_net, train_loaded_subsets, test_loaded_subsets, loss_fn, optimizer, n_classes, bag_weight,  num_epochs=1):
    
    since = time.time()
    #best_model_embedding_wts = copy.deepcopy(embedding_net.state_dict())
    #best_model_classification_wts = copy.deepcopy(classification_net.state_dict())
    best_auc = 0.

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        ##################################
        # TRAIN
        
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        inst_logger = Accuracy_Logger(n_classes=n_classes)
        
        train_loss = 0 # train_loss
        train_error = 0 
        train_inst_loss = 0.
        inst_count = 0
        
        train_acc = 0
        
        ###################################
        # TEST
        
        val_acc_logger = Accuracy_Logger(n_classes)
        val_inst_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_error = 0.
    
        val_inst_loss = 0.
        val_inst_count= 0
        
        val_acc = 0

        ###################################
        # TRAIN
        
        embedding_net.eval()
        classification_net.train(True)
        
        for batch_idx, loader in enumerate(train_loaded_subsets.values()):

            print("\rTraining batch {}/{}".format(batch_idx, len(train_loaded_subsets)), end='', flush=True)
            
            optimizer.zero_grad()
            
            patient_embedding = []
            for data in loader:
                
                inputs, label = data
                
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
                
                embedding = embedding_net(inputs)
                
                embedding = embedding.detach().to('cpu')
                embedding = embedding.squeeze(0)
                patient_embedding.append(embedding)
                
            patient_embedding = torch.stack(patient_embedding)
            patient_embedding = patient_embedding.cuda()
            
            logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            
            train_acc += torch.sum(Y_hat == label.data)
            
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            
            train_loss += loss_value
            if (batch_idx + 1) % 20 == 0:
                print('- batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                    'label: {}, bag_size: {}'.format(label.item(), patient_embedding.size(0)))
     
            error = classification_net.calculate_error(Y_hat, label)
            train_error += error
            
            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss /= len(train_loaded_subsets)
        train_error /= len(train_loaded_subsets)
        train_accuracy =  train_acc / len(train_loaded_subsets)
        
        if inst_count > 0:
            train_inst_loss /= inst_count
            print('\n')
            for i in range(n_classes):
                acc, correct, count = inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
         
            
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}, train_accuracy: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                  
        #embedding_net.train(False)
        classification_net.train(False)
        
        ################################
        # TEST
        
        embedding_net.eval()
        classification_net.eval()
        
        prob = np.zeros((len(test_loaded_subsets), n_classes))
        labels = np.zeros(len(test_loaded_subsets))
        #sample_size = classification_net.k_sample

        for batch_idx, loader in enumerate(test_loaded_subsets.values()):
            
            print("\rValidation batch {}/{}".format(batch_idx, len(test_loaded_subsets)), end='', flush=True)
            
            patient_embedding = []
            for data in loader:
                
                inputs, label = data
                
                with torch.no_grad():
                    if use_gpu:
                        inputs, label = inputs.cuda(), label.cuda()
                    else:
                        inputs, label = inputs, label
                
                embedding = embedding_net(inputs)
                embedding = embedding.detach().to('cpu')
                embedding = embedding.squeeze(0)
                patient_embedding.append(embedding)
                
            patient_embedding = torch.stack(patient_embedding)
            patient_embedding = patient_embedding.cuda()
            
            logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
            val_acc_logger.log(Y_hat, label)
            
            val_acc += torch.sum(Y_hat == label.data)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            val_inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            val_inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.detach().to('cpu').numpy()
            labels[batch_idx] = label.item()
            
            error = classification_net.calculate_error(Y_hat, label)
            val_error += error
            
            
        val_error /= len(test_loaded_subsets)
        val_loss /= len(test_loaded_subsets)
        val_accuracy = val_acc / len(test_loaded_subsets)
        
        if n_classes == 2:
            val_auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
        
            val_auc = np.nanmean(np.array(aucs))
        
        clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
                    
        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, val_auc, val_accuracy))
        if val_inst_count > 0:
            val_inst_loss /= val_inst_count
            for i in range(n_classes):
                acc, correct, count = val_inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    
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

        if val_auc > best_auc:
            best_model_classification_wts = copy.deepcopy(classification_net.state_dict())
            best_auc = val_auc
            
    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    classification_net.load_state_dict(best_model_classification_wts)
        
    return embedding_net, classification_net



def test_slides(embedding_net, classification_net, test_loaded_subsets, loss_fn, n_classes): 
               
    since = time.time()
    
    ###################################
    # TEST
    
    val_acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_count=0
    
    val_acc = 0
    
    incorrect_preds = []
    
    embedding_net.eval()
    classification_net.eval()
    
    prob = np.zeros((len(test_loaded_subsets), n_classes))
    labels = np.zeros(len(test_loaded_subsets))

    for batch_idx, loader in enumerate(test_loaded_subsets.values()):
        
        print("\rValidation batch {}/{}".format(batch_idx, len(test_loaded_subsets)), end='', flush=True)
        
        patient_embedding = []
        for data in loader:
            
            inputs, label = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
            
            embedding = embedding_net(inputs)
            embedding = embedding.detach().to('cpu')
            embedding = embedding.squeeze(0)
            patient_embedding.append(embedding)
            
        patient_embedding = torch.stack(patient_embedding)
        patient_embedding = patient_embedding.cuda()
        
        logits, Y_prob, Y_hat, _, instance_dict = classification_net(patient_embedding, label=label, instance_eval=True)
        val_acc_logger.log(Y_hat, label)
        
        val_acc += torch.sum(Y_hat == label.data)
        
        if not Y_hat == label.data:
            incorrect_preds.append(loader.dataset.filepaths[-1])
        
        loss = loss_fn(logits, label)

        val_loss += loss.item()

        instance_loss = instance_dict['instance_loss']
        
        val_inst_count+=1
        instance_loss_value = instance_loss.item()
        val_inst_loss += instance_loss_value

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        val_inst_logger.log_batch(inst_preds, inst_labels)

        prob[batch_idx] = Y_prob.detach().to('cpu').numpy()
        labels[batch_idx] = label.item()
        
        error = classification_net.calculate_error(Y_hat, label)
        val_error += error
        
        
    val_error /= len(test_loaded_subsets)
    val_loss /= len(test_loaded_subsets)
    val_accuracy = val_acc / len(test_loaded_subsets)
    
    if n_classes == 2:
        val_auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
    
        val_auc = np.nanmean(np.array(aucs))

    clsf_report = pd.DataFrame(classification_report(labels, np.argmax(prob, axis=1), output_dict=True, zero_division=1)).transpose()
    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, val_auc, val_accuracy))
    if val_inst_count > 0:
        val_inst_loss /= val_inst_count
        for i in range(2):
            acc, correct, count = val_inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))


    for i in range(n_classes):
        acc, correct, count = val_acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    print(clsf_report)
    print(conf_matrix)
    
    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1]) # TN / (TN + FP)
        print('Sensitivity: ', sensitivity) 
        print('Specificity: ', specificity)

    elapsed_time = time.time() - since
    
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        
    return val_error, val_auc, val_accuracy, val_acc_logger, labels, prob, clsf_report, conf_matrix, sensitivity, specificity, incorrect_preds



def soft_vote(vgg16, loaded_subsets):
    
    since = time.time()

    history = defaultdict(list)

    acc_train = 0

    ################################
    # SOFT VOTE
    
    vgg16.eval()
    
    #train_total = 0
    
    preds_x_class = []
    
    for i, loader in enumerate(loaded_subsets.values()):

        print("\rTesting batch {}/{}".format(i, len(loaded_subsets)), end='', flush=True)
        
        patient_soft_voting = []
        
        for data in loader:
            
            inputs, labels = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
        
            output = vgg16(inputs)
            probs = F.softmax(output, dim=1)
            np_probs = probs.detach().to('cpu')
            patient_soft_voting.append(np_probs)
            
        prob_x_class = torch.stack(patient_soft_voting).sum(axis=0)
        preds_x_class.append(prob_x_class / len(loader))
        max_prob_class = torch.argmax(prob_x_class)
        acc_train += (max_prob_class == labels).sum().item()
        
        history['actual'].append(labels.detach().to('cpu').numpy())
        history['predicted'].append(max_prob_class.detach().to('cpu').numpy())
                                
        del inputs, labels, output
        gc.collect()
        torch.cuda.empty_cache()
        
    avg_acc = acc_train / len(loaded_subsets)
    
    print()
    print("Avg acc: {:.4f}".format(avg_acc))
    print('-' * 10)
    print()

    elapsed_time = time.time() - since
    
    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    
    return history, patient_soft_voting


# %%