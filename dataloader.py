from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import SepsisDataset

# Ensure that all tensors in the torch.stack of the dataloader is of the same size to avoid dimension error
def padding(batch_list: list) -> tuple:

    # Separate the features from the labels for each patient in the batch
    batch_features = [patient[0] for patient in batch_list]
    batch_labels = [patient[1] for patient in batch_list]

    # Calculate the number of patient records for each patient in the batch to get original data back after padding
    lengths = torch.tensor([len(feature) for feature in batch_features], dtype=torch.long)

    # Add redundant 0.0 values to ensure each tensor has the same dimension
    padded_features = pad_sequence(batch_features, batch_first=True, padding_value=0.0)
    # padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0.0)

    patient_labels = torch.tensor([
        1.0 if (torch.tensor(l) == 1).any() else 0.0 for l in batch_labels
    ], dtype=torch.float32)

    return padded_features, patient_labels, lengths

def create_training_set() -> DataLoader:

    # Setup pytorch dataset with tensors
    home_dir = Path.home()
    input_dir = home_dir / "sepsis_training_files"
    Dataset = SepsisDataset(input_dir)

    # Setup DataLoader to create batches for training the neural network model
    DataLoaderInstance = DataLoader(Dataset, batch_size=32, shuffle=True, collate_fn=padding)

    return DataLoaderInstance