from pathlib import Path
from datasets import DataPreprocessor

def download_dataset(input_dir: Path, output_dir: Path) -> None:

    # Instantiate the preprocessor
    Preprocessor = DataPreprocessor()

    # Parse .psv files and convert to tensors
    Preprocessor.batch_process_files(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=False,
        max_workers=None
    )

if __name__ == "__main__":

    # Create cv set
    home_dir = Path.home()

    input_dirA = home_dir / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training" / "training_setA"
    output_dirA = home_dir / "sepsis_training_files"

    input_dirB = home_dir / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training" / "training_setB"
    output_dirB = home_dir / "sepsis_cv_and_test_files"

    download_dataset(input_dir=input_dirA, output_dir=output_dirA)
    download_dataset(input_dir=input_dirB, output_dir=output_dirB)