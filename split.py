import os
import shutil
from pathlib import Path
import random

def split_tif_files(source_folder, output_folder, train_ratio=0.8, test_ratio=0.10, valid_ratio=0.10, seed=42):
    """
    Split .tif files from source folder into train/test/valid folders.
    
    Args:
        source_folder: Path to folder containing .tif files
        output_folder: Path to output folder (will create train/test/valid subfolders)
        train_ratio: Proportion for training set (default 0.7)
        test_ratio: Proportion for test set (default 0.15)
        valid_ratio: Proportion for validation set (default 0.15)
        seed: Random seed for reproducibility
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all .tif files
    source_path = Path(source_folder)
    tif_files = list(source_path.glob("*.tif")) + list(source_path.glob("*.TIF"))
    
    if not tif_files:
        print("No .tif files found in the source folder!")
        return
    
    print(f"Found {len(tif_files)} .tif files")
    
    # Shuffle files
    random.shuffle(tif_files)
    
    # Calculate split indices
    total_files = len(tif_files)
    train_end = int(total_files * train_ratio)
    test_end = train_end + int(total_files * test_ratio)
    
    # Split files
    train_files = tif_files[:train_end]
    test_files = tif_files[train_end:test_end]
    valid_files = tif_files[test_end:]
    
    # Create output directories
    output_path = Path(output_folder)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    valid_dir = output_path / "valid"
    
    for directory in [train_dir, test_dir, valid_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective folders
    print(f"\nCopying files...")
    print(f"Train: {len(train_files)} files")
    for file in train_files:
        shutil.copy2(file, train_dir / file.name)
    
    print(f"Test: {len(test_files)} files")
    for file in test_files:
        shutil.copy2(file, test_dir / file.name)
    
    print(f"Valid: {len(valid_files)} files")
    for file in valid_files:
        shutil.copy2(file, valid_dir / file.name)
    
    print(f"\nDone! Files split into:")
    print(f"  {train_dir}")
    print(f"  {test_dir}")
    print(f"  {valid_dir}")

# Example usage
if __name__ == "__main__":
    source_folder = "./drawings/cropped"
    output_folder = "./split"
    
    # Standard 80/10/10 split
    split_tif_files(source_folder, output_folder, 
                    train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1)

    # Or use 80/10/10 split
    # split_tif_files(source_folder, output_folder, 
    #                 train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1)