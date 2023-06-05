# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:02:06 2023

@author: AmayaGS
"""

import os, os.path

from PIL import Image
from PIL import ImageFile

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

from loaders import Loaders
from clam_model import VGG_embedding
from Graph_model import GAT_SAGPool
use_gpu = torch.cuda.is_available()

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

if use_gpu:
    print("Using CUDA")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import gc
gc.enable()

# %%

file = r"C:\Users\Amaya\Documents\PhD\Data\df_all_stains_patches_labels.csv"
#file = "/data/scratch/wpw030/RA/df_all_stains_patches_labels_HPC.csv"

# %%

def run_hyperparameter(config, label='Pathotype_binary', patient_id="Patient ID", data_dir=file):

    torch.manual_seed(2)
    train_fraction = .7
    random_state = 2

    subset = True
    slide_batch = 1

    num_workers = 0
    shuffle = False

    train_slides = True
    testing_slides = True

    embedding_vector_size = 1024
    n_classes = 2

    K = 5
    epoch = 1

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

    df = pd.read_csv(file, header=0)
    df = df.dropna(subset=[label])

    file_ids, train_ids, test_ids = Loaders().train_test_ids(
        df, train_fraction, random_state, patient_id, label, subset=subset)

    CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST = Loaders(
    ).dictionary_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, slide_batch, num_workers)

    embedding_net = VGG_embedding(
        embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    graph_net = GAT_SAGPool(
        1024, heads=config["heads"], pooling_ratio=config["pooling_ratio"], dropout=config["dropout"])

    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(graph_net.parameters(),
                              lr=config["learning_rate"])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        graph_net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer_ft.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    for epoch in range(start_epoch, 10):
        ##################################
        # TRAIN

        train_loss = 0  # train_loss
        train_error = 0
        train_inst_loss = 0.
        inst_count = 0

        train_acc = 0

        train_count = 0

        ###################################
        # TEST

        val_loss = 0.
        val_acc = 0

        ###################################
        # TRAIN
        
        embedding_net.to(device)
        embedding_net.eval()
        graph_net.to(device)
        graph_net.train(True)

        for batch_idx, loader in enumerate(zip(CD138_patients_TRAIN.values(), CD68_patients_TRAIN.values(), CD20_patients_TRAIN.values(), HE_patients_TRAIN.values())):

            patient_embedding = []

            for i, data in enumerate(loader):

                slide_embedding = []

                for patch in data:

                    inputs, label = patch

                    inputs, label = inputs.to(device), label.to(device)
                    if torch.cuda.device_count() > 1:
                        embedding_net = nn.DataParallel(embedding_net)

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

            knn_graph = kneighbors_graph(
                patient_embedding, K, mode='connectivity', include_self=False)
            edge_index = torch.tensor(
                np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index).to(device)

            logits, Y_prob = graph_net(data)
            Y_hat = Y_prob.argmax(dim=1)
            loss = loss_fn(logits, label)

            train_acc += torch.sum(Y_hat == label.data)
            train_count += 1

            # backward pass
            loss.backward()
            # step
            optimizer_ft.step()
            optimizer_ft.zero_grad()

            del data, knn_graph, edge_index, slide_embedding, patient_embedding, logits, Y_prob, Y_hat
            gc.collect()

        total_loss = loss.item() / train_count
        train_accuracy = train_acc / train_count

        graph_net.train(False)
        
        

        ################################
        # TEST

        #embedding_net.eval()
        graph_net.eval()

        test_count = 0

        prob = []
        labels = []

        for batch_idx, loader in enumerate(zip(CD138_patients_TEST.values(), CD68_patients_TEST.values(), CD20_patients_TEST.values(), HE_patients_TEST.values())):

            patient_embedding = []

            for i, data in enumerate(loader):

                slide_embedding = []

                for patch in data:

                    inputs, label = patch

                    inputs, label = inputs.to(device), label.to(device)
                    if torch.cuda.device_count() > 1:
                        embedding_net = nn.DataParallel(embedding_net)

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

            knn_graph = kneighbors_graph(
                patient_embedding, K, mode='connectivity', include_self=False)
            edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index).to(device)

            logits, Y_prob = graph_net(data)
            Y_hat = Y_prob.argmax(dim=1)

            val_acc += torch.sum(Y_hat == label.data)
            test_count += 1

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            prob.append(Y_prob.detach().to('cpu').numpy())
            labels.append(label.item())

            del data, knn_graph, edge_index, slide_embedding, patient_embedding, logits, Y_prob, Y_hat
            gc.collect()

        val_loss /= test_count
        val_accuracy = val_acc / test_count
        val_acc = val_accuracy.detach().to('cpu').numpy()

        if n_classes == 2:
            prob = np.stack(prob, axis=1)[0]
            val_auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(
                labels, classes=[i for i in range(n_classes)])
            prob = np.stack(prob, axis=1)[0]
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(
                        binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            val_auc = np.nanmean(np.array(aucs))

        checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": graph_net.state_dict(),
        "optimizer_state_dict": optimizer_ft.state_dict(),
        }

        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
        {"loss": val_loss, "accuracy": val_acc, "AUC": val_auc},
        checkpoint=checkpoint,
        )

    del embedding_net, graph_net
    gc.collect()

    print("Finished")

# %%

def main(max_num_epochs=10, num_samples=10):

    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "heads": tune.choice([1, 2, 4, 6, 8, 10]),
        "pooling_ratio": tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.4, 0.6, 0.8])
    }

    scheduler = ASHAScheduler(
        max_t= max_num_epochs,
        grace_period= 1,
        reduction_factor= 2,
    )

    result = tune.run(
        tune.with_parameters(run_hyperparameter),
        verbose=1,
        config= config,
        scheduler= scheduler,
        metric="accuracy",
        mode="max",
        stop={"training_iteration": max_num_epochs},
        num_samples= num_samples,
        local_dir= "./ray_results",
        name="test_experiment"
    )

    best_trial = result.get_best_trial("AUC", "max", "all")
    best_checkpoint = result.get_best_checkpoint(best_trial, metric="AUC")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation AUC: {}".format(
        best_trial.last_result["AUC"]))

# %%

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(max_num_epochs=5, num_samples=1)

# %%    

    
# def main(max_num_epochs=10, num_samples=10):

#     config = {
#         "learning_rate": tune.loguniform(1e-5, 1e-1),
#         "heads": tune.choice([1, 2, 4, 6, 8, 10]),
#         "pooling_ratio": tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
#         "dropout": tune.choice([0, 0.1, 0.2, 0.4, 0.6, 0.8])
#     }

#     scheduler = ASHAScheduler(
#         max_t= max_num_epochs,
#         grace_period= 1,
#         reduction_factor= 2,
#     )
    
#     tune_config = tune.TuneConfig(metric="accuracy", mode="max", scheduler=scheduler, num_samples= num_samples)    
        
#     tuner = tune.Tuner(run_hyperparameter,
#                        param_space=config,
#                        tune_config= tune_config,
#                        run_config=air.RunConfig(name="test_experiment"))
    
#     result_grid = tuner.fit()

#     best_trial = result_grid.get_best_result(metric="accuracy", mode="max")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.get_best_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.get_best_result["accuracy"]))
#     print("Best trial final validation AUC: {}".format(
#         best_trial.get_best_result["AUC"]))

# %%