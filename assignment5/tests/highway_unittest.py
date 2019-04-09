import unittest
from highway import Highway
import torch

class TestHighway(unittest.TestCase):

    def setUp(self):
        self.x_conv = torch.tensor([[1.0,2.0,3.0],
                                    [4.0,5.0,6.0]])
        self.batch_size = 2
        self.embed_size = 3
        self.highway = Highway(self.embed_size, 0.5)

    def test_forward(self):
        result = self.highway.forward(self.x_conv)
        #print("W_proj:", self.highway.projection.weight)
        #print("b_proj:", self.highway.projection.bias)
        #print("W_gate:", self.highway.gate.weight)
        #print("b_gate:", self.highway.gate.bias)
        self.assertEqual(list(result.size()), [self.batch_size, self.embed_size], "Should be size of [batch_size, embed_size]")
        
if __name__ == '__main__':
    unittest.main()