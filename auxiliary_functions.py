# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:03:24 2023

@author: AmayaGS

"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class histoDataset(Dataset):

    def __init__(self, df, transform, label):
        
        self.transform = transform 
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['Location'].tolist()
        self.stain = df['Stain'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = Image.open(self.filepaths[idx])
        image_tensor = self.transform(image)
        image_label = self.labels[idx]
            
        return image_tensor, image_label     
    

class Accuracy_Logger(object):
    
    """Accuracy logger"""
    def __init__(self, n_classes):
        #super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count