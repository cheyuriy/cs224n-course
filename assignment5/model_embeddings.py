#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 
from vocab import VocabEntry, Vocab

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.highway = Highway(embed_size, 0.3)
        self.max_word_length = 21
        self.cnn = CNN(50, embed_size, 5)
        self.char_embeddings = nn.Embedding(len(vocab.char2id), 50, vocab.char2id['<pad>'])


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_emb = self.char_embeddings(input)
        x_reshaped = x_emb.transpose(2,3)
        x_reshaped_flatted = x_reshaped.reshape((-1, x_reshaped.size()[2], x_reshaped.size()[3]))
        x_conv_out = self.cnn.forward(x_reshaped_flatted)
        x_word_emb = self.highway.forward(x_conv_out)
        x_word_emb_reshaped = x_word_emb.reshape((x_emb.size()[0], x_emb.size()[1], x_word_emb.size()[1]))
        return x_word_emb_reshaped

        ### END YOUR CODE

