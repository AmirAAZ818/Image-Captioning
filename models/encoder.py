#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Encoder module for image captioning.
This module implements the encoder part of the image captioning system.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting feature representations from images.
    Uses a pre-trained CNN backbone with the classification head removed.
    """
    
    def __init__(self, model_name : str='resnet18', embed_size : int=256, pretrained : bool=True, trainable : bool=False):
        """
        Initialize the encoder.
        
        Args:
            model_name (str): Name of the CNN backbone to use
                Supported models: 'resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3'
            embed_size (int): Dimensionality of the output embeddings
            pretrained (bool): Whether to use pre-trained weights
            trainable (bool): Whether to fine-tune the CNN backbone
        """
        super(EncoderCNN, self).__init__()
        
        self.model_name = model_name.lower()
        self.embed_size = embed_size
        # TODO: Initialize and configure the CNN backbone based on model_name
        # 1. Create the CNN model using torchvision.models with pretrained weights if specified
        # 2. Store the feature dimension size (before the final classifier)
        # 3. Remove the classifier/fully-connected layer and replace with nn.Identity()
        # Hint: Look at model architectures to find the classifier attribute name and output feature size
        self.cnn, self.feature_size = self._prepare_cnn(pretrained=pretrained)
        
        # TODO: Create a projection layer to transform CNN features to embed_size
        # The projection should include normalization, activation, and regularization
        self.projection = nn.Sequential(
            nn.Linear(self.feature_size, self.embed_size),
            nn.BatchNorm1d(self.embed_size),
            nn.ReLU(),
            nn.Dropout(0.3)            
        )
        
        # TODO: Freeze CNN parameters if trainable is False
        # This prevents the CNN backbone from being updated during training
        if not trainable:
            self._freeze_cnn()
    def _freeze_cnn(self):
        """
        Prevents the CNN backbone from being updated during training
        """
        for param in self.cnn.parameters():
            param.requires_grad = False
            
    def _prepare_cnn(self, pretrained) -> Tuple[nn.Module, int]:
        """
        Extracts CNN backbone model and its feature size.
        
        Returns:
            Tuple[nn.Module, int]: CNN backbone, feature size
        """
        model = None
        feature_size = None
        if self.model_name == 'resnet50':
            # Model adaptation
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            model.fc = nn.Identity()
            
            # Feature size declaration
            feature_size = 2048
        elif self.model_name == 'resnet18':
            # Model adaptation
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Identity()
            
            # Feature size declaration
            feature_size = 512
        elif self.model_name == 'mobilenet_v2':
            # Model adaptation
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
            model.classifier = nn.Identity()
            
            # Feature size declaration
            feature_size = 1280
        elif self.model == 'inception_v3':
            # Model adaptation
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Identity()
            
            # Feature size declaration
            feature_size = 2048
        else:
            raise Exception(f"Model name does not exist: {self.model_name}")
        
        return model, feature_size
    
    def forward(self, images):
        """
        Forward pass to extract features from images.
        
        Args:
            images (torch.Tensor): Batch of input images [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Image features [batch_size, embed_size]
        """
        # Extract features from CNN
        features = self.cnn(images)
        
        # Project features to the specified embedding size
        features = self.projection(features)
        
        return features
    
    def get_feature_size(self):
        """Returns the raw feature size of the CNN backbone"""
        return self.feature_size