# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:34:24 2023

@author: AmayaGS
"""

import time
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch

from auxiliary_functions import Accuracy_Logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc
gc.enable()

def train_graph_multi_wsi(graph_net, train_loader, test_loader, loss_fn, optimizer, n_classes, num_epochs=1, checkpoint=True, checkpoint_path="PATH_checkpoints"):


    since = time.time()
    best_acc = 0.
    best_AUC = 0.

    results_dict = {}

    train_loss_list = []
    train_accuracy_list = []
    #train_auc_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []

    for epoch in range(num_epochs):

        ##################################
        # TRAIN
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        train_acc = 0
        train_count = 0
        graph_net.train()

        print("Epoch {}/{}".format(epoch, num_epochs), flush=True)
        print('-' * 10)

        for batch_idx, (patient_ID, graph_object) in enumerate(train_loader.dataset.items()):

            data, label = graph_object

            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

            logits, Y_prob = graph_net(data)
            Y_hat = Y_prob.argmax(dim=1)
            acc_logger.log(Y_hat, label)
            train_loss = loss_fn(logits, label)

            train_acc += torch.sum(Y_hat == label.data)
            train_count += 1

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del data, logits, Y_prob, Y_hat
            gc.collect()

        total_loss = train_loss.item() / train_count
        train_accuracy =  train_acc / train_count

        train_loss_list.append(total_loss)
        train_accuracy_list.append(train_accuracy.item())

        print()
        print('Epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(epoch, total_loss, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        ################################
        # TEST/EVAL
        graph_net.eval()

        val_acc_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_acc = 0
        val_count = 0

        prob = []
        labels = []

        for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

            data, label = graph_object

            with torch.no_grad():
                if use_gpu:
                    data, label = data.cuda(), label.cuda()
                else:
                    data, label = data, label

            logits, Y_prob = graph_net(data)
            Y_hat = Y_prob.argmax(dim=1)
            val_acc_logger.log(Y_hat, label)

            val_acc += torch.sum(Y_hat == label.data)
            val_count += 1

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            prob.append(Y_prob.detach().to('cpu').numpy())
            labels.append(label.item())

            del data, logits, Y_prob, Y_hat
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

        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

        print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_auc, val_accuracy), flush=True)

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

                if checkpoint:
                    checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                    torch.save(graph_net.state_dict(), checkpoint_weights)

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    if checkpoint:
        graph_net.load_state_dict(torch.load(checkpoint_weights), strict=True)

    results_dict = {'train_loss': train_loss_list,
                    'val_loss': val_loss_list,
                    'train_accuracy': train_accuracy_list,
                    'val_accuracy': val_accuracy_list,
                    'val_auc': val_auc_list
                    }

    return graph_net, results_dict


# TEST

def test_graph_multi_wsi(graph_net, test_loader, loss_fn, n_classes=2):

    since = time.time()

    test_acc_logger = Accuracy_Logger(n_classes)
    test_loss = 0.
    test_acc = 0
    test_count = 0

    prob = []
    labels = []
    test_loss_list = []
    test_accuracy_list = []
    test_auc_list = []

    graph_net.eval()

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        data, label = graph_object

        with torch.no_grad():
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

        logits, Y_prob = graph_net(data)
        Y_hat = Y_prob.argmax(dim=1)
        test_acc_logger.log(Y_hat, label)

        test_acc += torch.sum(Y_hat == label.data)
        test_count += 1

        loss = loss_fn(logits, label)
        test_loss += loss.item()

        prob.append(Y_prob.detach().to('cpu').numpy())
        labels.append(label.item())

        del data, logits, Y_prob, Y_hat
        gc.collect()

    test_loss /= test_count
    test_accuracy = test_acc / test_count

    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy.item())

    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
        test_auc = roc_auc_score(labels, prob[:, 1])
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

        test_auc = np.nanmean(np.array(aucs))

    test_auc_list.append(test_auc)

    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

    print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_auc, test_accuracy), flush=True)

    print(conf_matrix)

    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        print('Sensitivity: ', sensitivity)
        print('Specificity: ', specificity)

    elapsed_time = time.time() - since

    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return labels, prob, conf_matrix, sensitivity, specificity