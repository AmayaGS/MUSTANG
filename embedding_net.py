# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:08:28 2023

@author: AmayaGS
"""

import torch.nn as nn
from torchvision import models


class VGG_embedding(nn.Module):

    """
    VGG16 embedding network for WSI patches
    """

    def __init__(self, embedding_vector_size=1024, n_classes=2):

        super(VGG_embedding, self).__init__()

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

        features = list(embedding_net.classifier.children())[:-2] # Remove last layer
        embedding_net.classifier = nn.Sequential(*features)
        self.vgg_embedding = nn.Sequential(embedding_net)

    def forward(self, x):

        output = self.vgg_embedding(x)
        output = output.view(output.size()[0], -1)
        return output