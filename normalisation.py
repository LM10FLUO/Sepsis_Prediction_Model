
import torch
from pathlib import Path

def calc_mean_and_std() -> None:

    # Retrieve the directory storing preprocessed training tensors
    home_dir = Path.home()
    training_dir = home_dir / "sepsis_training_files_6h"

    all_rows = []

    # Collate all patient files into a single list for processing
    for file_path in sorted(training_dir.glob("*.pt")):
        patient_tensor = torch.load(file_path)
        all_rows.append(patient_tensor["x"])

    # Flatten the list into a 2d array for efficient operations
    all_rows = torch.cat(all_rows, dim=0)

    # Calculate the mean and standard deviations
    mean = all_rows.mean(dim=0)
    std = all_rows.std(dim=0)

    # Prevent divide by 0 errors by replacing std = 0 to 1
    std = torch.where(std==0, torch.ones_like(std), std)

    torch.save({"mean": mean, "std": std}, "feature_norm_stats.pt")

if __name__ == "__main__":

    calc_mean_and_std()