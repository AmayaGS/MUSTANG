# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:02 2022

@author: AmayaGS
"""

import os
import os.path
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import pickle
import argparse

from PIL import Image
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# MUSTANG functions
from loaders import Loaders
from embedding_net import VGG_embedding
from create_store_graphs import create_embeddings_graphs
from graph_train_loop import train_graph_multi_wsi
from auxiliary_functions import seed_everything
from Graph_model import GAT_SAGPool
from plotting_results import auc_plot, pr_plot, confusion_matrix_plot, learning_curve_plot


def main():

    parser = argparse.ArgumentParser(description="Multi-stain self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--dataset_name", type=str, default="RA", help="Dataset name")
    parser.add_argument("--PATH_patches", type=str, default="df_labels.csv", help="CSV file with patch file location")
    parser.add_argument("--embedding_vector_size", type=int, default=1024, help="Embedding vector size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--pooling_ratio", type=float, default=0.7, help="Pooling ratio")
    parser.add_argument("--heads", type=int, default=2, help="Number of GAT heads")
    parser.add_argument("--K", type=int, default=5, help="Number of nearest neighbours in k-NNG created from WSI embeddings")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction")
    #parser.add_argument("--slide_batch", type=int, default=10, help="Slide batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--graph_batch_size", type=int, default=1, help="Graph batch size for training")
    parser.add_argument("--checkpoint", action="store_false", default=True, help="Enable checkpointing of GNN weights. Set to False if you don't want to store checkpoints.")

    args = parser.parse_args()

    # Set environment variables
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Check for GPU availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set image properties
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    plt.ion()

    # Image transforms
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

    # Parameters
    seed = args.seed
    seed_everything(seed)
    train_fraction = args.train_fraction
    subset= False
    slide_batch = 10 # this needs to be larger than one, otherwise Dataloader can fail when only passed a None object from collate function. TODO. change Dataset to Iterable dataset to solve this problem.
    K= args.K
    num_workers = args.num_workers
    batch_size = args.graph_batch_size
    creating_knng = False
    creating_embedding = False
    train_graph = True
    embedding_vector_size = args.embedding_vector_size
    learning_rate = args.learning_rate
    pooling_ratio = args.pooling_ratio
    heads = args.heads
    num_epochs = args.num_epochs

    label = 'label'
    patient_id = 'Patient ID'
    dataset_name = args.dataset_name
    n_classes= args.n_classes

    checkpoint = args.checkpoint
    current_directory = Path(__file__).resolve().parent
    results = os.path.join(current_directory, f"results/graph_{dataset_name}_{seed}_{heads}_{pooling_ratio}_{learning_rate}")
    os.makedirs(results, exist_ok = True)

   # Load the dataset
    df = pd.read_csv(args.PATH_patches, header=0)
    df = df.dropna(subset=[label])

    # Define collate function
    def collate_fn_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # create k-NNG with VGG patch embedddings
    if creating_knng:
        file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, seed, patient_id, label, subset)
        train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=subset)
        train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
        embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
        if use_gpu:
            embedding_net.cuda()
        # Save k-NNG with VGG patch embedddings for future use
        slides_dict = {('train_graph_dict_', 'train_embedding_dict_') : train_slides ,
                       ('test_graph_dict_', 'test_embedding_dict_'): test_slides}
        for file_prefix, slides in slides_dict.items():
            graph_dict, embedding_dict = create_embeddings_graphs(embedding_net, slides, k=K, mode='connectivity', include_self=False)
            print(f"Started saving {file_prefix[0]} to file")
            with open(f"{file_prefix[0]}{dataset_name}.pkl", "wb") as file:
                pickle.dump(graph_dict, file)  # encode dict into Pickle
                print("Done writing graph dict into pickle file")
            print(f"Started saving {file_prefix[1]} to file")
            with open(f"{file_prefix[1]}{dataset_name}.pkl", "wb") as file:
                pickle.dump(embedding_dict, file)  # encode dict into Pickle
                print("Done writing embedding dict into pickle file")

    # load pickled embeddings and graphs
    if not creating_knng:
        with open(f"train_graph_dict_{dataset_name}.pkl", "rb") as train_file:
        # Load the dictionary from the file
            train_graph_dict = pickle.load(train_file)
        with open(f"test_graph_dict_{dataset_name}.pkl", "rb") as test_file:
        # Load the dictionary from the file
            test_graph_dict = pickle.load(test_file)

    if not creating_embedding:
        with open(f"train_embedding_dict_{dataset_name}.pkl", "rb") as train_file:
        # Load the dictionary from the file
            train_embedding_dict = pickle.load(train_file)
        with open(f"test_embedding_dict_{dataset_name}.pkl", "rb") as test_file:
        # Load the dictionary from the file
            test_embedding_dict = pickle.load(test_file)

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

    # MULTI-STAIN GRAPH
    train_graph_loader = torch.utils.data.DataLoader(train_graph_dict, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, drop_last=False)
    #train_graph_loader = torch_geometric.loader.DataLoader(train_graph_dict, batch_size=1, shuffle=False, num_workers=0, sampler=sampler, drop_last=False, generator=seed_everything(state)) #TODO MINIBATCHING
    test_graph_loader = torch.utils.data.DataLoader(test_graph_dict, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    if train_graph:
        graph_net = GAT_SAGPool(embedding_vector_size, heads=heads, pooling_ratio=pooling_ratio)
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(graph_net.parameters(), lr=learning_rate)
        if use_gpu:
            graph_net.cuda()

        graph_weights, results_dict = train_graph_multi_wsi(graph_net, train_graph_loader, test_graph_loader, loss_fn, optimizer_ft, n_classes, num_epochs=num_epochs, checkpoint=checkpoint, checkpoint_path= results + "/checkpoint_")

        torch.save(graph_weights.state_dict(), results + ".pth")

        df_results = pd.DataFrame.from_dict(results_dict)
        df_results.to_csv(results + ".csv", index=False)
        learning_curve_plot(df_results)

# %%

if __name__ == "__main__":
    main()