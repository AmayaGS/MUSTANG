# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:15:48 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random

import numpy as np

from sklearn.model_selection import train_test_split

import torch

from auxiliary_functions import histoDataset


class Loaders:
    
    #def __init__(self):
    

    def df_loader(self, df, train_transform, test_transform, train_fraction, random_state, patient_id, label, subset=False):
        
        # patients need to be strictly separated between splits to avoid leakage. 
        ids  = df[patient_id].tolist()
        file_ids = sorted(set(ids))
    
        train_ids, test_ids = train_test_split(file_ids, test_size=1-train_fraction, random_state=random_state)
        train_sub = df[df[patient_id].isin(train_ids)].reset_index(drop=True)
        test_sub = df[df[patient_id].isin(test_ids)].reset_index(drop=True)
        df_train = histoDataset(train_sub, train_transform, label=label)
        df_test = histoDataset(test_sub, test_transform, label=label)
        
        if subset:
            train_subset_ids = random.sample(train_ids, 5)
            test_subset_ids = random.sample(test_ids, 3)
            train_sub_sample = df[df[patient_id].isin(train_subset_ids)].reset_index(drop=True)
            test_sub_sample = df[df[patient_id].isin(test_subset_ids)].reset_index(drop=True)
            df_train_sample = histoDataset(train_sub_sample, train_transform, label=label)
            df_test_sample = histoDataset(test_sub_sample, test_transform, label=label)
            
            return df_train_sample, df_test_sample, train_sub_sample, test_sub_sample, file_ids, train_subset_ids, test_subset_ids
        
        return df_train, df_test, train_sub, test_sub, file_ids, train_ids, test_ids
    
    
    def patches_dataloader(self, df_train, df_test, sampler, train_batch, test_batch, num_workers, shuffle, drop_last, collate_fn):
        
        train_loader = torch.utils.data.DataLoader(df_train, batch_size=train_batch, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(df_test, batch_size=test_batch, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

        return train_loader, test_loader


    def slides_dataloader(self, train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, label='Binary disease', patient_id="Patient ID"):
        
        # TRAIN dict
        train_subsets = {}
        file_indices = []
        
        for i, file in enumerate(train_ids):
            file_indices.append(np.where(train_sub["Patient ID"] == file))
            train_subsets['subset_%02d' % i] = histoDataset(train_sub[file_indices[i][0][0]: file_indices[i][0][-1] + 1], train_transform, label=label)
        
        train_loaded_subsets = {}
        
        for i, value in enumerate(train_subsets.values()):
            train_loaded_subsets['subset_%02d' % i] = torch.utils.data.DataLoader(value, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)
            
        # TEST dict
        test_subsets = {}
        file_indices = []
        
        for i, file in enumerate(test_ids):
            file_indices.append(np.where(test_sub["Patient ID"] == file))
            test_subsets['subset_%02d' % i] = histoDataset(test_sub[file_indices[i][0][0]: file_indices[i][0][-1] + 1], test_transform, label=label)
        
        test_loaded_subsets = {}
        
        for i, value in enumerate(test_subsets.values()):
            test_loaded_subsets['subset_%02d' % i] = torch.utils.data.DataLoader(value, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)
            
        return train_loaded_subsets, test_loaded_subsets
    
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
            