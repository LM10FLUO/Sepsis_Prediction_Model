from pathlib import Path
from sklearn.model_selection import train_test_split
from dataloader import create_dataset
from neural_network import PredictionModel
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    home_dir = Path.home()
    test_file_dir = home_dir / "sepsis_cv_and_test_files"

    testing_files = sorted(test_file_dir.glob("*.pt"))

    # We are using the same random_state as in training so that the test files remain distinct from the cv files
    cv_set, test_set = train_test_split(testing_files, test_size=0.2, random_state=12)

    test_dataloader = create_dataset(dataset=test_set)

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
    all_labels = []

    with torch.no_grad():
        for features, labels, lengths in test_dataloader:
            features, labels = features.to(device), labels.to(device).float()

            outputs = Model(features, lengths)
            probs = torch.sigmoid(outputs.squeeze(-1))

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().tolist())

    test_auroc = roc_auc_score(all_labels, all_probs)
    test_auprc = average_precision_score(all_labels, all_probs)

    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    # Let us set the threshold for what we define as positive and negative be 0.5
    # This means any probability flagged as 0.5 or more will be considered at high risk to sepsis

    threshold = 0.3
    preds = (np.array(all_probs) >= threshold).astype(int)

    # Generate a confusion matrix to visualise flase positives and negatives in the test set
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold = {threshold})")
    plt.show()