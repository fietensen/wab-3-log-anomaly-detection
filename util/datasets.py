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

    (cldt_bgl_train, ) = _read_transform_cldt_bgl(bgl_path)

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
            "bgl": cldt_bgl_train,
        #    "hdfs": DataLoader(cldt_hdfs_dataset, batch_size=hdfs_batch_size, pin_memory=True)
        }
    }


def _read_transform_cldt_bgl(bgl_path: Path, device: torch.device = torch.device("cpu")) -> tuple[DataLoader]:
    "Reads the BG/L Dataset from the specified directory"

    bgl_dataset_preprocessed = bgl_path / "preprocessed.cldt.h5"

    with h5py.File(bgl_dataset_preprocessed, "r") as fp:
        norm_iids = torch.tensor(np.array(fp["data"]["normative"]["input_ids"][:512], dtype=np.uint16)).to(device, dtype=torch.uint16)
        norm_ams = torch.tensor(np.array(fp["data"]["normative"]["attention_mask"][:512], dtype=np.uint8)).to(device, dtype=torch.uint8)
        norm_labels = torch.zeros((norm_iids.shape[0], 1)).to(device, dtype=torch.uint8)

        anom_iids = torch.tensor(np.array(fp["data"]["anomalous"]["input_ids"][:512], dtype=np.uint16)).to(device, dtype=torch.uint16)
        anom_ams = torch.tensor(np.array(fp["data"]["anomalous"]["attention_mask"][:512], dtype=np.uint8)).to(device, dtype=torch.uint8)
        anom_labels = torch.ones((anom_iids.shape[0], 1)).to(device, dtype=torch.uint8)

    # TODO: introduce train, val, test split
    ds_iids = torch.vstack((norm_iids, anom_iids))
    del norm_iids
    del anom_iids

    ds_ams = torch.vstack((norm_ams, anom_ams))
    del norm_ams
    del anom_ams

    ds_lbls = torch.vstack((norm_labels, anom_labels))
    del norm_labels
    del anom_labels

    dataset = TripletDataset(ds_iids, ds_ams, ds_lbls)
    sampler = TripletBatchSampler(dataset, 32)
    dataloader = DataLoader(dataset, batch_sampler=sampler)#, pin_memory=True)

    return (dataloader,)


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
