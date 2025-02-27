from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset
from autolog_processor import process_logfile, process_bgl_line, process_hdfs_line

import pandas as pd
import os


def read_datasets_make_dataloader(bgl_path: Path, hdfs_path: Path, /, bgl_batch_size: int = 32, hdfs_batch_size: int = 32) -> dict[str, DataLoader]:
    """
    Reads the BG/L and HDFS_v1 Datasets from Disk and returns DataLoader
    that can be used for training.

    Arguments:
        - bgl_path (Path): Path to the BGL Dataset directory
        - hdfs_path (Path): Path to the HDFS Dataset directory
        (optional) bgl_batch_size (int): Batch size to use in the DataLoader for the BG/L Dataset
        (optional) hdfs_batch_size (int): Batch size to use in the DataLoader for the HDFS Dataset
    """

    bgl_dataset = _read_transform_bgl(bgl_path)
    hdfs_dataset = _read_transform_bgl(hdfs_path)

    return {
        "bgl": DataLoader(bgl_dataset, batch_size=bgl_batch_size, pin_memory=True),
        "hdfs": DataLoader(hdfs_dataset, batch_size=hdfs_batch_size, pin_memory=True)
    }


def _read_transform_bgl(bgl_path: Path) -> Dataset:
    "Reads the BG/L Dataset from the specified directory"

    bgl_log_file = bgl_path / "BGL.log"
    bgl_log_parsed = bgl_path / "BGL.prep.csv"

    parsed = None

    if os.path.isfile(bgl_log_parsed):
        print("[*] Loading cached preprocessed file from " + bgl_log_parsed)
        parsed = pd.read_csv(bgl_log_parsed)

    else:
        parsed = process_logfile(
            bgl_log_file,
            line_processor=process_bgl_line,
            print_progress=True
        )
        parsed.to_csv(bgl_log_parsed)

    print(parsed.head())


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
