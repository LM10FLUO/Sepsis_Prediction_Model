from dataloader import create_training_set
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import PredictionModel

if __name__ == "__main__":

    dataloader = create_training_set()

    # The pos_weight is the ratio of negative to positive (0 / 1) examples in the dataset
    # This essentially helps to scale the number of sepsis onset cases to reduce imbalance in the dataset
    pos_weight = 10.0
    SepsisModel = PredictionModel(input_size=40,
                                  hidden_size=64,
                                  num_layers=1,
                                  out_features=1
                                  )

    # Apply weighting to loss function to scale the loss from incorrectly identifying sepsis cases
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimiser = optim.Adam(SepsisModel.parameters(), lr=0.001)

    # Use GPU if available, else fall back to CPU usage and moves all model weights, params and buffers onto selected hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SepsisModel.to(device)

    # Train the LSTM Model

    num_epochs = 100
    running_loss_history = []

    for epoch in range(num_epochs):
        # Sets the model in training mode
        SepsisModel.train()
        running_loss = 0.0

        # Iterate through each batch in the data loader
        for i, (features, labels, lengths) in enumerate(dataloader):
            # Load features and labels onto the dedicated hardware device
            features, labels = features.to(device), labels.to(device).float()

            # Forward propogate through the model and calculate the loss
            outputs = SepsisModel(features, lengths)
            loss = loss_function(outputs.squeeze(-1), labels.squeeze(-1))

            # Backpropogate through the model to update the weights and biases in the model
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        running_loss_history.append(round(running_loss / len(dataloader), 4))