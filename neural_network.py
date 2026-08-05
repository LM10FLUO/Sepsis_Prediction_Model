import torch
import torch.nn as nn
import torch.optim as optim
from datasets import SepsisDataset

# 

# Creating the neural network
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(41, 32),           # n.b. the Linear function is the linear regression function wx + b
            nn.Sigmoid(),                # Here it compresses the input tensor size of size 41 into an output tensor of size 32
            nn.Linear(32, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward_prop(self, x):
        return self.model