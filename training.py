from dataloader import create_training_set, create_cv_set
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import PredictionModel
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from modelling import display_loss
import matplotlib.pyplot as plt
from random import randint
from math import inf

if __name__ == "__main__":

    home_dir = Path.home()
    test_file_dir = home_dir / "sepsis_cv_and_test_files"

    testing_files = sorted(test_file_dir.glob("*.pt"))

    # Here we are going to use an 80/20 split for CV size to test size
    cv_set, test_set = train_test_split(testing_files, test_size=0.2, random_state=12)

    training_dataloader = create_training_set()
    cv_dataloader = create_cv_set(cv_set=cv_set)

    # The pos_weight is the ratio of negative to positive (0 / 1) examples in the dataset
    # This essentially helps to scale the number of sepsis onset cases to reduce imbalance in the dataset
    pos_weight = 3.0
    SepsisModel = PredictionModel(input_size=40,
                                  hidden_size=64,
                                  num_layers=1,
                                  out_features=1,
                                  dropout_prob=0.5
                                  )

    # Use GPU if available, else fall back to CPU usage and moves all model weights, params and buffers onto selected hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SepsisModel.to(device)

    # Apply weighting to loss function to scale the loss from incorrectly identifying sepsis cases
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device)

    # Use regularisation to help reduce overfitting
    optimiser = optim.Adam(SepsisModel.parameters(), lr=0.0001, weight_decay=0.001)

    # Train the LSTM Model

    num_epochs = 100
    epoch = 1
    train_loss_history = []
    cv_loss_history = []
    early_stop = False

    # If the CV loss shows consistent divergence, early stop training to save best general model
    best_cv_loss = float(inf)
    patience_counter = 0
    max_patience = 15

    while epoch <= num_epochs and not early_stop:

        print(f"Epoch [{epoch}/{num_epochs}]")

        # Sets the model in training mode
        SepsisModel.train()
        train_running_loss = 0.0
        cv_running_loss = 0.0

        random_training = randint(0, len(training_dataloader)-1)
        random_cv = randint(0, len(cv_dataloader)-1)

        # Iterate through each batch in the data loader
        for i, (features, labels, lengths) in enumerate(training_dataloader):

            # Load features and labels onto the dedicated hardware device
            features, labels = features.to(device), labels.to(device).float()

            # Forward propogate through the model and calculate the loss
            outputs = SepsisModel(features, lengths)
            train_loss = loss_function(outputs.squeeze(-1), labels.squeeze(-1))

            # Backpropogate through the model to update the weights and biases in the model
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

            if i == random_training:
                print("\nTRAINING:\n")
                print(f"{'Label':<8}{'Prediction':<12}")
                for label, prob in zip(labels.tolist(), torch.sigmoid(outputs).squeeze(-1).tolist()):
                    print(f"{label:<8.0f}{prob:<12.4f}")

            train_running_loss += train_loss.item()

        # sets the model in evaluation mode to test the cross validation set for generalisation of the model
        SepsisModel.eval()

        with torch.no_grad():
            for j, (features, labels, lengths) in enumerate(cv_dataloader):
                features, labels = features.to(device), labels.to(device).float()

                outputs = SepsisModel(features, lengths)
                cv_loss = loss_function(outputs.squeeze(-1), labels.squeeze(-1))

                if j == random_cv:
                    print("\nCROSS VALIDATION\n")
                    print(f"{'Label':<8}{'Prediction':<12}")
                    for label, prob in zip(labels.tolist(), torch.sigmoid(outputs).squeeze(-1).tolist()):
                        print(f"{label:<8.0f}{prob:<12.4f}")

                cv_running_loss += cv_loss.item()

        avg_train_loss = train_running_loss / len(training_dataloader)
        avg_cv_loss = cv_running_loss / len(cv_dataloader)

        # If the current state has a better cv loss than previously, save the current model state
        if avg_cv_loss <= best_cv_loss:
            best_cv_loss = avg_cv_loss
            torch.save(SepsisModel.state_dict(), "best_model_checkpoint.pt")
            patience_counter = 0

        # If the cv loss diverges, check whether max_patience is reached for early stopping
        else:
            patience_counter += 1

            if patience_counter >= max_patience:
                early_stop = True
                print(f"Early stopping at epoch {epoch+1}. Best CV loss: {best_cv_loss:.4f}")

        print(f"Training Loss: {avg_train_loss:.4f}, CV Loss: {avg_cv_loss:.4f}\n")
        print("____________________________________________________________________")
        train_loss_history.append(avg_train_loss)
        cv_loss_history.append(avg_cv_loss)

        epoch += 1

    display_loss(train_loss_history, "Training loss vs epochs")
    display_loss(cv_loss_history, "CV loss vs epochs")

    plt.show()