import torch
import torch.nn as nn
import torch.optim as optim
from datasets import SepsisDataset
from torch.nn.utils.rnn import pack_padded_sequence

"""
LSTM Model Theory:
The LSTM Model builds on the RNN model, addressing its key issue: the vanishing/exploding gradient problem. As the RNN
model involves a feedback loop that uses a weight, this causes this to exponentially decrease or increase, affecting
gradient descent. The LSTM model uses 3 inputs: the long term memory input (cell state), short term memory input (hidden 
state) and the input. Instead of relying on a feedback loop, the percentage of long term memory coupled with the 
short term memory ensures we can use previous data to influence future predictions without the issue seen with RNNs
"""

class PredictionModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
            Pack padded tensors to remove redundant data, compressing tensors in each batch into a 1D array alongside
            the original tensor size so the computer can correctly "decompress" the input data
        """
        x_padded = pack_padded_sequence(input=x, lengths=lengths.cpu(), batch_first=True)

        # Forward propogate input through the neural network
        h_n_collection, (h_n, c_n) = self.lstm(x_padded)
        out = h_n[-1]
        out = self.fc(out)
        out = self.sigmoid(out)

        return out