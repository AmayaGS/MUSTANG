# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:15:48 2023

@author: AmayaGS
"""

import os, os.path
import numpy as np
from PIL import Image
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class histoDataset(Dataset):

    def __init__(self, df, transform, label):
        
        self.transform = transform 
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['Location'].tolist()
        self.stain = df['Stain'].tolist()
        self.patient_ID = df['Patient ID'].tolist()
        self.filename = df['Filename'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = Image.open(self.filepaths[idx])
        patient_id = self.patient_ID[idx]
        filename = self.filename[idx]
        stain = self.stain[idx]
        image_tensor = self.transform(image)
        image_label = self.labels[idx]
            
        return image_tensor, image_label, patient_id, filename, stain

class Loaders:
    
    #def __init__(self):
        
    def train_test_ids(self, df, train_fraction, random_state, patient_id, label, subset=False):
        
        # patients need to be strictly separated between splits to avoid leakage. 
        ids  = df[patient_id].tolist()
        file_ids = sorted(set(ids))
    
        train_ids, test_ids = train_test_split(file_ids, test_size=1-train_fraction, random_state=random_state)
        
        if subset:
            
            train_subset_ids = random.sample(train_ids, 10)
            test_subset_ids = random.sample(test_ids,5)
            
            return file_ids, train_subset_ids, test_subset_ids
        
        return file_ids, train_ids, test_ids

    def df_loader(self, df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False):
        
        train_subset = df[df[patient_id].isin(train_ids)].reset_index(drop=True)
        test_subset = df[df[patient_id].isin(test_ids)].reset_index(drop=True)
        df_train = histoDataset(train_subset, train_transform, label=label)
        df_test = histoDataset(test_subset, test_transform, label=label)
        
        return df_train, df_test, train_subset, test_subset


    def slides_dataloader(self, train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, label='Pathotype_binary', patient_id="Patient ID"):
        
        # TRAIN dict
        train_subsets = {}
        for i, file in enumerate(train_ids):
            new_key = f'{file}'
            train_subset = histoDataset(train_sub[train_sub["Patient ID"] == file], train_transform, label=label)
#            if len(train_subset) != 0:
            train_subsets[new_key] = torch.utils.data.DataLoader(train_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)

            
        # TEST dict
        test_subsets = {}
        for i, file in enumerate(test_ids):
            new_key = f'{file}'
            test_subset = histoDataset(test_sub[test_sub["Patient ID"] == file], test_transform, label=label)
#            if len(test_subset) != 0:
            test_subsets[new_key] = torch.utils.data.DataLoader(test_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)
        
        return train_subsets, test_subsets


    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
            