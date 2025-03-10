from model_eval.tr_eval_autolog import train_autolog
from model_eval.tr_eval_cldtlog import train_cldtlog

from util.datasets import read_datasets_make_dataloader
from pathlib import Path

# rng modules for ensuring reproducibility
import random
import numpy as np
import torch

def get_metrics(model, dataset: dict) -> dict[str, float]:
    r_norm = model.classify(dataset["test_normal"]).astype(int)
    r_anom = model.classify(dataset["test_anomalous"]).astype(int)

    # False Positives
    FP = r_norm.sum()

    # True Negatives
    TN = r_norm.shape[0] - FP

    # True Positives
    TP = r_anom.sum()

    # False Negatives
    FN = r_anom.shape[0] - TP

    return {
        "f1": (2*TP)/(2*TP + FP + FN),
        "precision": TP/(TP+FP) if TP+FP else 0,
        "recall": TP/(TP+FN)
    }

if __name__ == '__main__':
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)

    bgl_path = Path(r".\datasets\BGL")
    hdfs_path = Path(r".\datasets\HDFS_v1")

    datasets = read_datasets_make_dataloader(
        bgl_path=bgl_path,
        hdfs_path=hdfs_path
    )

    
    al_bgl_classifier = train_autolog(datasets["autolog"]["bgl"])
    al_bgl_metrics = get_metrics(al_bgl_classifier, datasets["autolog"]["bgl"])
    print("AutoLog - Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
        al_bgl_metrics["precision"],
        al_bgl_metrics["recall"],
        al_bgl_metrics["f1"]
    ))
    

    cldtlog_bgl_classifier = train_cldtlog(datasets["cldtlog"]["bgl"], epochs=10)
    cldtlog_bgl_metrics = get_metrics(cldtlog_bgl_classifier, datasets["cldtlog"]["bgl"])
    print("CLDTLog - Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
        cldtlog_bgl_metrics["precision"],
        cldtlog_bgl_metrics["recall"],
        cldtlog_bgl_metrics["f1"]
    ))
