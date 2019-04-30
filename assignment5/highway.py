#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """ Highway network implementation"""

    def __init__(self, embed_size, dropout_rate):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param dropout_rate (float): Dropout probability
        """

        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        self.projection = nn.Linear(embed_size, embed_size, bias = True)
        self.relu = nn.ReLU()
        self.gate = nn.Linear(embed_size, embed_size, bias = True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of output from convolutional layer and compute embeddings.

        @param batch (Tensor): list of outputs from convolutional network. Size: (b, embed_size)

        @returns words_embeds (Tensor): a variable/tensor of shape (b, embed_size) representing the
                                    embedding for each word in minibatch
        """
        p = self.relu(self.projection(batch))
        g = self.sigmoid(self.gate(batch))
        words_embeds = self.dropout(g*p + (1 - g)*batch)

        return words_embeds

### END YOUR CODE 

