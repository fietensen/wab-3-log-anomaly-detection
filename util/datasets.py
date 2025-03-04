from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset
from util.autolog_preprocessor import process_logfile, process_bgl_line, process_hdfs_line

import pandas as pd


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

    al_bgl_dataset, cldt_bgl_dataset = _read_transform_bgl(bgl_path)
    al_hdfs_dataset, cldt_hdfs_dataset = _read_transform_bgl(hdfs_path)

    return {
        "autolog": {
            "bgl": DataLoader(al_bgl_dataset, batch_size=bgl_batch_size, pin_memory=True),
            "hdfs": DataLoader(al_hdfs_dataset, batch_size=hdfs_batch_size, pin_memory=True)
        },
        "cldtlog": {
            "bgl": DataLoader(cldt_bgl_dataset, batch_size=bgl_batch_size, pin_memory=True),
            "hdfs": DataLoader(cldt_hdfs_dataset, batch_size=hdfs_batch_size, pin_memory=True)
        }
    }


def _read_transform_bgl(bgl_path: Path) -> Dataset:
    "Reads the BG/L Dataset from the specified directory"

    bgl_log_file = bgl_path / "preprocessed.al.csv"

    return None, None


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
