from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from util.tripletdatset import TripletDataset, TripletBatchSampler

import pandas as pd
import numpy as np
import torch
import h5py


def read_datasets_make_dataloader(bgl_path: Path, hdfs_path: Path, *, bgl_batch_size: int = 32, hdfs_batch_size: int = 32) -> dict[str, dict]:
    """
    Reads the BG/L and HDFS_v1 Datasets from Disk and returns DataLoader
    that can be used for training.

    Arguments:
        - bgl_path (Path): Path to the BGL Dataset directory
        - hdfs_path (Path): Path to the HDFS Dataset directory
        (optional) bgl_batch_size (int): Batch size to use in the DataLoader for the BG/L Dataset
        (optional) hdfs_batch_size (int): Batch size to use in the DataLoader for the HDFS Dataset
    """

    (al_bgl_train, al_bgl_val, al_bgl_test_norm, al_bgl_test_anom) = _read_transform_al_bgl(bgl_path)
    (al_hdfs_train, al_hdfs_val, al_hdfs_test_norm, al_hdfs_test_anom) = _read_transform_al_hdfs(hdfs_path)

    # determine device based on cuda presence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (cldt_bgl_train, cldt_bgl_val, cldt_bgl_test_norm, cldt_bgl_test_anom) = _read_transform_cldt_bgl(bgl_path, device=device)

    return {
        "autolog": {
            "bgl": {
                "train": DataLoader(al_bgl_train, batch_size=bgl_batch_size, pin_memory=True),
                "val": al_bgl_val,
                "test_normal": al_bgl_test_norm,
                "test_anomalous": al_bgl_test_anom
            },
            "hdfs": {
                "train": DataLoader(al_hdfs_train, batch_size=hdfs_batch_size, pin_memory=True),
                "val": al_hdfs_val,
                "test_normal": al_hdfs_test_norm,
                "test_anomalous": al_hdfs_test_anom
            }
        },
        "cldtlog": {
            "bgl": {
                "train": cldt_bgl_train,
                "val": cldt_bgl_val,
                "test_normal": cldt_bgl_test_norm,
                "test_anomalous": cldt_bgl_test_anom
            }
        #    "hdfs": DataLoader(cldt_hdfs_dataset, batch_size=hdfs_batch_size, pin_memory=True)
        }
    }


def _read_transform_cldt_bgl(bgl_path: Path, device: torch.device = torch.device("cpu")) -> tuple[DataLoader]:
    "Reads the BG/L Dataset from the specified directory"

    bgl_dataset_preprocessed = bgl_path / "preprocessed.cldt.h5"

    with h5py.File(bgl_dataset_preprocessed, "r") as fp:
        norm_iids = np.array(fp["data"]["normative"]["input_ids"][:30355], dtype=np.uint16)
        norm_ams = np.array(fp["data"]["normative"]["attention_mask"][:30355], dtype=np.uint8)

        anom_iids = np.array(fp["data"]["anomalous"]["input_ids"][:2640], dtype=np.uint16)
        anom_ams = np.array(fp["data"]["anomalous"]["attention_mask"][:2640], dtype=np.uint8)

    # Distribution: Train: 60% | Val: 20% | Test: 20%
    tsplit_norm_iids, test_norm_iids, tsplit_norm_ams, test_norm_ams = train_test_split(norm_iids, norm_ams, test_size=0.2)
    train_norm_iids, val_norm_iids, train_norm_ams, val_norm_ams = train_test_split(tsplit_norm_iids, tsplit_norm_ams, test_size=0.25)

    tsplit_anom_iids, test_anom_iids, tsplit_anom_ams, test_anom_ams = train_test_split(anom_iids, anom_ams, test_size=0.2)
    train_anom_iids, val_anom_iids, train_anom_ams, val_anom_ams = train_test_split(tsplit_anom_iids, tsplit_anom_ams, test_size=0.25)

    train_iids = torch.vstack((
        torch.tensor(train_norm_iids).to(device, dtype=torch.uint16),
        torch.tensor(train_anom_iids).to(device, dtype=torch.uint16),
    ))

    train_ams = torch.vstack((
        torch.tensor(train_norm_ams).to(device, dtype=torch.uint8),
        torch.tensor(train_anom_ams).to(device, dtype=torch.uint8)
    ))

    train_labels = torch.vstack((
        torch.zeros((train_norm_iids.shape[0], 1), device=device, dtype=torch.uint8),
        torch.ones((train_anom_iids.shape[0], 1), device=device, dtype=torch.uint8),
    ))

    val_iids = torch.vstack((
        torch.tensor(val_norm_iids).to(device, dtype=torch.uint16),
        torch.tensor(val_anom_iids).to(device, dtype=torch.uint16)
    ))

    val_ams = torch.vstack((
        torch.tensor(val_norm_ams).to(device, dtype=torch.uint8),
        torch.tensor(val_anom_ams).to(device, dtype=torch.uint8)
    ))

    val_labels = torch.vstack((
        torch.zeros((val_norm_iids.shape[0], 1), device=device, dtype=torch.uint8),
        torch.ones((val_anom_iids.shape[0], 1), device=device, dtype=torch.uint8),
    ))

    train_dataset = TripletDataset(train_iids, train_ams, train_labels)
    train_sampler = TripletBatchSampler(train_dataset, 64, anom_ratio=0.08)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    val_dataset = TripletDataset(val_iids, val_ams, val_labels)
    val_sampler = TripletBatchSampler(val_dataset, 64, deterministic=True, anom_ratio=0.08)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

    test_norm = (
        torch.tensor(test_norm_iids).to(device, dtype=torch.int64),
        torch.tensor(test_norm_ams).to(device, dtype=torch.uint8)
    )

    test_anom = (
        torch.tensor(test_anom_iids).to(device, dtype=torch.int64),
        torch.tensor(test_anom_ams).to(device, dtype=torch.uint8)
    )

    return (train_dataloader, val_dataloader, test_norm, test_anom)


def _read_transform_al_bgl(bgl_path: Path) -> tuple[Dataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    "Reads the BG/L Dataset from the specified directory"

    bgl_dataset_preprocessed = bgl_path / "preprocessed.al.csv"
    bgl_df = pd.read_csv(bgl_dataset_preprocessed).drop("tblk", axis=1)

    bgl_df_norm = bgl_df[bgl_df["anomalous"] == False].drop("anomalous", axis=1).to_numpy()
    bgl_df_anom = bgl_df[bgl_df["anomalous"] == True].drop("anomalous", axis=1).to_numpy()

    x_train, x_test = train_test_split(bgl_df_norm, test_size=0.2)

    # further split train data into train and validation
    x_train, x_val = train_test_split(x_train, test_size=0.1)

    scaler = MinMaxScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
    
    x_test_norm = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    x_test_anom = torch.tensor(scaler.transform(bgl_df_anom), dtype=torch.float32)

    return (TensorDataset(x_train), x_val, x_test_norm, x_test_anom)


def _read_transform_al_hdfs(hdfs_path: Path) -> tuple[Dataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    "Reads the HDFS Dataset from the specified directory"

    hdfs_dataset_preprocessed = hdfs_path / "preprocessed.al.csv"
    hdfs_df = pd.read_csv(hdfs_dataset_preprocessed).drop("tblk", axis=1)

    hdfs_df_norm = hdfs_df[hdfs_df["anomalous"] == False].drop("anomalous", axis=1).to_numpy()
    hdfs_df_anom = hdfs_df[hdfs_df["anomalous"] == True].drop("anomalous", axis=1).to_numpy()

    x_train, x_test = train_test_split(hdfs_df_norm, test_size=0.2)

    # further split train data into train and validation
    x_train, x_val = train_test_split(x_train, test_size=0.1)

    scaler = MinMaxScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
    
    x_test_norm = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    x_test_anom = torch.tensor(scaler.transform(hdfs_df_anom), dtype=torch.float32)

    return (TensorDataset(x_train), x_val, x_test_norm, x_test_anom)


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
