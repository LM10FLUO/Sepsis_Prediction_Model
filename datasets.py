import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

class DataPreprocessor:

    # Load baseline values stored in external JSON file
    with open("baselines.json", mode="r", encoding="utf-8") as baselines_file:
        baselines = json.load(baselines_file)

    @classmethod
    def preprocessing(cls, df: pd.DataFrame, hours_ahead: int) -> pd.DataFrame:

        # Forward fill to remove NaN values, replacing them with the last reading recorded (assuming it has not changed since)      
        df = df.ffill()

        # Replace remaining NaN values with baseline "healthy" values
        row_indices, col_indices = np.where(df.isna())

        for row_index, col_index in zip(row_indices ,col_indices):
            baseline = df.columns[col_index]

            if baseline in cls.baselines:
                df.iloc[row_index, col_index] = cls.baselines[baseline]

        # Add target labels to the data and remove additional records after the first target onset of sepsis

        target_series = df["SepsisLabel"].shift(-hours_ahead)
        df["Target"] = target_series.ffill().fillna(0).astype(int)

        df.drop(df[df["SepsisLabel"] == 1].index, inplace=True)

        return df

    # Saving a single file as a pytorch tensor
    @staticmethod
    def process_single_file(file_args: tuple) -> bool:
    
        input_path, output_dir, overwrite = file_args
        output_path = output_dir / f"{input_path.stem}.pt"
    
        # If the file has already been cleaned and converted and we do not want to overwrite, avoid wasting time reprocessing the file
        if output_path.exists() and not overwrite:
            return True

        try:
            # Clean the chosen file
            df = pd.read_csv(input_path, sep="|")
            df_cleaned = DataPreprocessor.preprocessing(df=df, hours_ahead=12)

            # Do not save empty cleaned patient records - i.e. patients who have sepsis on admission - to avoid errors
            if len(df_cleaned) == 0:
                print(f"{input_path.name} contains empty records")
                return False
    
            # Separate the data into the input features and target outputs
            input_features = df_cleaned.drop(columns=["SepsisLabel", "Target"]).to_numpy()
            target_outputs = df_cleaned[["Target"]].to_numpy()
    
            input_tensor = torch.from_numpy(input_features.copy()).float()
            target_tensor = torch.from_numpy(target_outputs.copy()).float()
    
            # Save the tensors in a dictionary alongside some metadata
            torch.save({"x": input_tensor, "y": target_tensor, "patient_id": input_path.stem}, output_path)
            return True

        except Exception as e:
            print(f"Failed to process file {input_path.name}: {e}")

            # If the file is partially written, delete the corrupted file
            if output_path.exists():
                output_path.unlink()
                
            return False

    @staticmethod
    def batch_process_files(input_dir: Path, output_dir: Path, overwrite: bool, max_workers: int) -> bool:

        # If the output directory does not exist, create a new directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Return a list of all the .psv files in the input directory to iterate through
        input_files = list(input_dir.iterdir())
        
        # Generate input tuples to feed into the process_single_file function
        file_args = [(input_path, output_dir, overwrite) for input_path in input_files]

        # Parallelise the process_single_file function across the max cores available (max_workers)
        # Use tqdm to show live progress bar to user in terminal
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            conversion_success = tqdm(
                executor.map(_process_file_worker, file_args),
                total=len(file_args),
                desc="Converting .psv files into .pt files"
            )

        for success in conversion_success:

            if not success:
                print("A file failed to process")
                return False

        print("Finished upload")
        return True

        # for parameters in file_args:
        #     success = DataPreprocessor.process_single_file(parameters)

        #     if not success:
        #         print("A file failed to process")
        #         return False
            
        # return True

# Standalone worker function at module level so background processes can find it
def _process_file_worker(args) -> bool:
    input_path, output_dir, overwrite = args
    return DataPreprocessor.process_single_file((input_path, output_dir, overwrite))

# Dataset class to enable batch processing of tensors via torch.utils.data.DataLoader
class SepsisDataset(Dataset):

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.input_files = sorted(self.input_dir.glob("*.pt"))

    # Returns number of files
    def __len__(self) -> int:
        return len(self.input_files)

    # Returns a single tensor
    def __getitem__(self, index: int) -> tuple:
        file_path = self.input_files[index]

        # Load the tensor dictionary stored and return the feature and target stored
        data = torch.load(file_path)
        return data["x"], data["y"]
        
            
        
        