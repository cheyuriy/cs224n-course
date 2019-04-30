import unittest
from cnn import CNN
from vocab import VocabEntry
import torch

class TestCNN(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.embed_size = 7
        self.x_emb = torch.ones((self.batch_size,4,12)) #[batch_size, char_embed_size, max_word_length]
        
        self.cnn = CNN(4,self.embed_size,3) #[char_embed_size, embed_size, kernel_size]

    def test_forward(self):
        result = self.cnn.forward(self.x_emb)
        self.assertEqual(list(result.size()), [self.batch_size, self.embed_size], "Should be size of [batch_size, embed_size]")
        
if __name__ == '__main__':
    unittest.main()