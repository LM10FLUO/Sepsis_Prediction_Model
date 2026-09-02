from pathlib import Path
from dataloader import create_dataset
from neural_network import PredictionModel
import torch


if __name__ == "__main__":

    file_path = input("Input file directory: ")
    file_path= Path(file_path)

    test_dataloader = create_dataset(dataset=[file_path])

    Model = PredictionModel(input_size=40,
                                hidden_size=64,
                                num_layers=1,
                                out_features=1,
                                dropout_prob=0.5
                                )

    # Load the saved weights and biases 
    Model.load_state_dict(torch.load("best_model_checkpoint.pt"))

    # Use GPU if available, else fall back to CPU usage and moves all model weights, params and buffers onto selected hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)
    Model.eval()

    all_probs = []

    with torch.no_grad():
        for features, labels, lengths in test_dataloader:
            features = features.to(device)

            outputs = Model(features, lengths)
            probs = torch.sigmoid(outputs.squeeze(-1))

            all_probs.extend(probs.cpu().tolist())

    if len(all_probs) == 1:
        print(f"The likelihood of this patient developing sepsis is: {round(all_probs[0] * 100, 2)}%")
    else:
        print(f"Warning: expected 1 prediction, got {len(all_probs)}. Predictions: {all_probs}")