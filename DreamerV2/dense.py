import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class DenseModel(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dist):
        super().__init__()
        self.activation = nn.ELU
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.dist = dist

        self.dense = nn.Sequential(
                        nn.Linear(self.in_size, self.hidden_size), self.activation(),
                        nn.Linear(self.hidden_size, self.hidden_size), self.activation(), 
                        nn.Linear(self.hidden_size, self.hidden_size), self.activation(),
                        nn.Linear(self.hidden_size, self.hidden_size), self.activation(),
                        nn.Linear(self.hidden_size, self.out_size[0])
                    )
        
    def forward(self, x):
        x = self.dense(x)
        #reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(x, 1), len(self.out_size))
        if self.dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=x), len(self.out_size))
        if self.dist == None:
            return x

