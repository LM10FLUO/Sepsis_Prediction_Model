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

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, out_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
            Pack padded tensors to remove redundant data, compressing tensors in each batch into a 1D array alongside
            the original tensor size so the computer can correctly "decompress" the input data
        """
        x_packed = pack_padded_sequence(input=x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)

        h_n_collection, (h_n, c_n) = self.lstm(x_packed)
        out = h_n[-1]
        out = self.fc1(out)

        return out