#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ Convolutional network implementation"""

    def __init__(self, input_channels, filters_n, kernel_size):
        """ Init CNN Model.

        @param input_channels (int): Number of channel in input
        @param kernel_size (int): Size of convolutional kernel
        @param filters_n (int): Number of filters in convolutional layer
        """

        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.filters_n = filters_n

        self.conv1d = nn.Conv1d(input_channels, filters_n, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of output from convolutional layer an compute embeddings.

        @param batch (Tensor): list of char embeddings from convolutional network. Size: (b, embed_size)

        @returns words_embeds (Tensor): a variable/tensor of shape (b, embed_size) representing the
                                    embedding for each word in minibatch
        """
        t = self.conv1d(batch)
        t = self.relu(t)
        t = F.max_pool1d(t, t.size()[2]) #we need to pass kernel_size equal to the last dimension of feature map
        conv_out = t.squeeze(2)
        return conv_out

### END YOUR CODE

