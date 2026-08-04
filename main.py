from pathlib import Path
import os
from datasets import DataPreprocessor, SepsisDataset

if __name__ == "__main__":

    home_dir = Path.home()

    # Locate the path directories to the .psv files from physionet.org and create a new output directory to save tensors
    input_dir = home_dir / "physionet.org" / "files/challenge-2019" / "1.0.0" / "training" / "training_setA"
    output_dir = home_dir / "sepsis_training_files"

    # Instantiate the preprocessor
    Preprocessor = DataPreprocessor()

    # Parse .psv files and convert to tensors
    Preprocessor.batch_process_files(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=False,
        max_workers=None
    )