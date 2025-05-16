#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full caption model integrating encoder and decoder.
This module combines the CNN encoder and RNN decoder into a complete image captioning model.
"""

import torch
import torch.nn as nn
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

class CaptionModel(nn.Module):
    """
    Complete image captioning model with CNN encoder and RNN decoder.
    """
    
    def __init__(self, 
             embed_size=256, 
             hidden_size=512, 
             vocab_size=10000, 
             num_layers=1,
             encoder_model='resnet18',
             decoder_type='lstm',
             dropout=0.5,
             train_encoder=False):
        """
        Initialize the caption model.
        
        Args:
            embed_size (int): Dimensionality of the embedding space
            hidden_size (int): Dimensionality of the RNN hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of layers in the RNN
            encoder_model (str): Name of the CNN backbone for the encoder
            decoder_type (str): Type of RNN cell ('lstm' or 'gru')
            dropout (float): Dropout probability
            train_encoder (bool): Whether to fine-tune the encoder
        """
        super(CaptionModel, self).__init__()
        
        # TODO: Initialize the encoder and decoder components
        # 1. Create an EncoderCNN instance with the specified parameters
        self.encoder = EncoderCNN(
            model_name=encoder_model,
            embed_size=embed_size,
            trainable=train_encoder,
        )
        
        # 2. Create a DecoderRNN instance with the specified parameters
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            rnn_type=decoder_type,
            dropout=dropout
        )
        
    def forward(self, images, captions, hidden=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images (torch.Tensor): Input images [batch_size, 3,hight, width]
            captions (torch.Tensor): Ground truth captions [batch_size, seq_length]
            hidden (tuple or torch.Tensor, optional): Initial hidden state for the RNN
            
        Returns:
            torch.Tensor: Output scores for each word in the vocabulary
                        Shape: [batch_size, seq_length, vocab_size]
            tuple or torch.Tensor: Final hidden state of the RNN
        """
        # TODO: Implement the forward pass of the full model
        # 1. Extract features from images using the encoder
        features = self.encoder(images)
        # 2. Use the decoder to generate captions based on the features and ground truth captions
        outputs, hidden = self.decoder(features, captions, hidden)
        
        # 3. Return the outputs and final hidden state
        return outputs, hidden
    
    def generate_caption(self, image: torch.Tensor, max_length=20, start_token=1, end_token=2, beam_size=1, temperature=1.) -> torch.Tensor:
        """
        Generate a caption for a single image.
        
        Args:
            image (torch.Tensor): Input image [1, 3, height, width]
            max_length (int): Maximum caption length
            start_token (int): Index of the start token
            end_token (int): Index of the end token
            beam_size (int): Beam size for beam search (1 = greedy search)
            
        Returns:
            torch.Tensor: Generated caption token sequence [1, seq_length]
        """
        sampled_ids = list()
        # TODO: Implement caption generation for inference
        # 1. Extract features from the image using the encoder (with torch.no_grad())
        with torch.no_grad():
            features = self.encoder(image)
        # 2. Use the decoder to generate a caption based on the features
            sampled_ids = self.decoder.sample(
                features=features,
                max_length=max_length,
                start_token=start_token,
                end_token=end_token,
                beam_size=beam_size,
                temperature=temperature
            )
        # 3. Return the generated caption
        return sampled_ids[0]  # Return first (and only) sequence in the batch