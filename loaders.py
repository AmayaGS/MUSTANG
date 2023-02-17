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
    
    
    def patches_dataloader(self, df_train, df_test, sampler, train_batch, test_batch, num_workers, shuffle, drop_last, collate_fn):
        
        train_loader = torch.utils.data.DataLoader(df_train, batch_size=train_batch, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, sampler=sampler, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(df_test, batch_size=test_batch, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

        return train_loader, test_loader


    def slides_dataloader(self, train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, label='Binary disease', patient_id="Patient ID"):
        
        # TRAIN dict
        train_subsets = {}
        #file_indices = []
        
        for i, file in enumerate(train_ids):
            new_key = f'{file}'
            train_subset = histoDataset(train_sub[train_sub["Patient ID"] == file], train_transform, label=label)
            train_subsets[new_key] = torch.utils.data.DataLoader(train_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)

            
        # TEST dict
        test_subsets = {}
        
        for i, file in enumerate(test_ids):
            new_key = f'{file}'
            test_subset = histoDataset(test_sub[test_sub["Patient ID"] == file], test_transform, label=label)
            test_subsets[new_key] = torch.utils.data.DataLoader(test_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False)
            
        return train_subsets, test_subsets
    
    def stain_dictionaries(train_loaded_subsets, test_loaded_subsets, stains):
        
        for stain in stains:
            
            new_key = f'{stain}'
            
            stain_patient_DataLoaders_TRAIN = {}
            stain_patient_DataLoaders_TEST = {}
            stain_patient_DataLoaders_TRAIN[new_key] = train_loaded_subsets
            stain_patient_DataLoaders_TEST[new_key] = test_loaded_subsets
    
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
    
        return CD138_patients_TRAIN, CD68_patients_TRAIN, CD20_patients_TRAIN, HE_patients_TRAIN, CD138_patients_TEST, CD68_patients_TEST, CD20_patients_TEST, HE_patients_TEST
        
        
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
            