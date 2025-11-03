#!/usr/bin/env python3
"""Download small dummy datasets for local testing."""

from src.core.datasets import download_dummy_dataset

if __name__ == "__main__":
    # Download dummy C4 dataset for training
    download_dummy_dataset(dataset_type="c4", num_samples=100, output_dir="data")

    # Download dummy C4 dataset for validation
    download_dummy_dataset(dataset_type="c4", num_samples=50, output_dir="data_eval")

    print("✓ Dummy datasets downloaded successfully!")
