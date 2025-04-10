from model_eval.tr_eval_autolog import train_autolog
from model_eval.tr_eval_cldtlog import train_cldtlog

from util.datasets import read_datasets_make_dataloader
from pathlib import Path
from typing import Callable
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

# rng modules for ensuring reproducibility
import random
import numpy as np
import torch

def get_metrics(model, dataset: dict, chunked_evaluation_fn: Callable | None = None, **kwargs) -> dict[str, float]:
    if chunked_evaluation_fn:
        r_norm = chunked_evaluation_fn(model, dataset["test_normal"], **kwargs).astype(int)
        r_anom = chunked_evaluation_fn(model, dataset["test_anomalous"], **kwargs).astype(int)
    else:
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
        "F1": (2*TP)/(2*TP + FP + FN),
        "Precision": TP/(TP+FP) if TP+FP else 0,
        "Recall": TP/(TP+FN)
    }

def chunked_cldtlog_evaluate(model, dataset, batch_size: int = 100, **kwargs) -> np.array:
    input_ids, input_ams = dataset
    results = np.zeros((len(input_ids), 1), dtype=np.float16)
    for i in tqdm(range(0, len(input_ids), batch_size)):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_input_ams = input_ams[i : i + batch_size]

        results[i : i + batch_size] = model.classify((batch_input_ids, batch_input_ams))
    
    return results


def get_model_throughput(model: torch.nn.Module, data: torch.Tensor | tuple[torch.Tensor, ...], iterations: int = 1, device: torch.DeviceObjType | None = None) -> int:
    if device != None:
        model = model.to(device)
        if type(data) == tuple:
            data = tuple([d.to(device) for d in data])
        else:
            data = data.to(device)

    t1 = time.time()
    for _ in range(iterations):
        model.classify(data)
    time_acc = (time.time()-t1) * 1000

    return time_acc/iterations


def log_metrics(model_name: str, metrics: dict) -> None:
    print("{} - Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
        model_name,
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"]
    ))


def plot_metrics(model_metrics: dict[str, dict[str, float]], outname: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    df = pd.DataFrame(list(model_metrics.values()), index=list(model_metrics.keys()))
    df_melted = df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
    df_melted.rename(columns={'index': 'Model'}, inplace=True)
    p = sns.barplot(df_melted, ax=ax, x="Metric", y="Value", hue="Model", width=0.5, gap=0.1)
    p.bar_label(p.containers[0], fontsize=10, labels=[f"{v:.2f}" for v in model_metrics["AutoLog"].values()])
    p.bar_label(p.containers[1], fontsize=10, labels=[f"{v:.2f}" for v in model_metrics["CLDTLog"].values()])
    p.legend_.remove()
    fig.legend(title="Model", loc="outside right upper")
    ax.set_title("Metric Comparison")

    if outname:
        fig.savefig(outname)
        print("[INFO] Wrote Figure to " + str(outname))
    else:
        plt.show() # blocking display


def plot_throughput(model_throughput: dict[str, dict[str, float]], log_scale: bool = True, outname: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    df = pd.DataFrame(list(model_throughput.values()), index=list(model_throughput.keys()))
    df_melted = df.reset_index().melt(id_vars="index", var_name="Device", value_name="Value")
    df_melted.rename(columns={'index': 'Model'}, inplace=True)
    p = sns.barplot(df_melted, ax=ax, x="Device", y="Value", hue="Model", width=0.5, gap=0.1)
    p.bar_label(p.containers[0], fontsize=10, labels=[f"{v:.3f}ms" for v in model_throughput["AutoLog"].values()])
    p.bar_label(p.containers[1], fontsize=10, labels=[f"{v:.3f}ms" for v in model_throughput["CLDTLog"].values()])
    p.legend_.remove()
    fig.legend(title="Model", loc="outside right upper")
    ax.set_title("Throughput Comparison{}".format(" (Log Scale)" if log_scale else ""))
    ax.set_ylabel("Time per Sample in Milliseconds")
    if log_scale:
        ax.set_yscale('log')

    if outname:
        fig.savefig(outname)
        print("[INFO] Wrote Figure to " + str(outname))
    else:
        plt.show() # blocking display


def plot_loss_curve(epoch_data: dict[str, list[float]], outname: str | Path | None = None, model_name: str | None = None) -> None:
    fig, ax = plt.subplots()
    epochs = np.arange(1, len(epoch_data["train_loss"]) + 1)
    ax.plot(epochs, epoch_data["train_loss"], label="Training")
    ax.plot(epochs, epoch_data["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs{}".format(" of " + model_name if model_name else ""))
    ax.legend(title="Dataset", loc="upper right")

    if outname:
        fig.savefig(outname)
        print("[INFO] Wrote Figure to " + str(outname))
    else:
        plt.show()


def plot_al_kde(df: pd.DataFrame, les: list[str], outname: str | Path | None = None) -> None:
    lepr = int(np.ceil(np.sqrt(len(les))))
    fig, ax = plt.subplots(lepr, lepr)
    for i, le in enumerate(les):
        f_ax = ax[i//lepr][i%lepr]
        f_ax.set_title(le)
        anomalous = df[(df["anomalous"] == 1) & (df[le] >= 1.0)][le].values
        normative = df[(df["anomalous"] == 0) & (df[le] >= 1.0)][le].values
        if i == 0:
            sns.kdeplot(anomalous, color="red", fill=True, alpha=0.5, ax=f_ax, label="Anomalous")
            sns.kdeplot(normative, color="blue", fill=True, alpha=0.5, ax=f_ax, label="Normative")
        else:
            sns.kdeplot(anomalous, color="red", fill=True, alpha=0.5, ax=f_ax)
            sns.kdeplot(normative, color="blue", fill=True, alpha=0.5, ax=f_ax)
        f_ax.set_xlim(0, 10)
        f_ax.set_xlabel("Score")
    fig.suptitle("KDEPlot of Logging-Entity Scores")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    fig.tight_layout()

    if outname:
        fig.savefig(outname)
        print("[INFO] Wrote Figure to " + str(outname))
    else:
        plt.show()


if __name__ == '__main__':
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)

    cuda = torch.device("cuda")

    bgl_path = Path(r".\datasets\BGL")

    datasets = read_datasets_make_dataloader(
        bgl_path=bgl_path,
        bgl_batch_size=2048
    )

    # KDEPlot showing normative/anomalous scores obtained from AutoLog preprocessing
    al_logging_entities = ["R00", "R01", "R02", "R03", "R04", "R05", "R06", "R10", "R11"]
    al_scores_df = pd.read_csv(bgl_path / "preprocessed.al.csv")
    plot_al_kde(al_scores_df, al_logging_entities, outname=r".\figures\autolog_scores_kde.png")

    al_bgl_training = {"train_loss": [], "val_loss": []}
    al_bgl_classifier = train_autolog(
        datasets["autolog"]["bgl"],
        log_val_loss=al_bgl_training["val_loss"],
        log_train_loss=al_bgl_training["train_loss"]
    )
    al_bgl_metrics = get_metrics(al_bgl_classifier, datasets["autolog"]["bgl"])
    log_metrics("AutoLog", al_bgl_metrics)

    cldtlog_bgl_training = {"train_loss": [], "val_loss": []}
    cldtlog_bgl_classifier = train_cldtlog(
        datasets["cldtlog"]["bgl"],
        epochs=10,
        log_val_loss=cldtlog_bgl_training["val_loss"],
        log_train_loss=cldtlog_bgl_training["train_loss"]
    )
    cldtlog_bgl_metrics = get_metrics(cldtlog_bgl_classifier, datasets["cldtlog"]["bgl"], chunked_evaluation_fn=chunked_cldtlog_evaluate)
    log_metrics("CLDTLog", cldtlog_bgl_metrics)

    # compare metrics
    plot_metrics({
        "AutoLog": al_bgl_metrics,
        "CLDTLog": cldtlog_bgl_metrics
    }, r".\figures\model_metric_comparison.png")

    # loss curves
    plot_loss_curve(al_bgl_training, model_name="AutoLog", outname=r".\figures\autolog_loss_curve.png")
    plot_loss_curve(cldtlog_bgl_training, model_name="CLDTLog", outname=r".\figures\cldtlog_loss_curve.png")

    # model throughput
    dataset_size = len(datasets["autolog"]["bgl"]["val"])
    al_throughput_cuda = get_model_throughput(al_bgl_classifier, datasets["autolog"]["bgl"]["val"], iterations=5_000, device=cuda) / dataset_size
    al_throughput_cpu = get_model_throughput(al_bgl_classifier, datasets["autolog"]["bgl"]["val"], iterations=5_000, device=torch.device("cpu")) / dataset_size

    dataset_size = 5
    dataset_input_ids, dataset_attention_masks = datasets["cldtlog"]["bgl"]["test_normal"]
    cldtlog_throughput_cuda = get_model_throughput(cldtlog_bgl_classifier, (dataset_input_ids[:5], dataset_attention_masks[:5]), iterations=50, device=cuda) / dataset_size
    cldtlog_throughput_cpu = get_model_throughput(cldtlog_bgl_classifier, (dataset_input_ids[:5], dataset_attention_masks[:5]), iterations=50, device=torch.device("cpu")) / dataset_size

    print("Model Throughput:")
    print("GPU:")
    print(f"AutoLog: {al_throughput_cuda:.4}ms / Sample")
    print(f"CLDTLog: {cldtlog_throughput_cuda:.4}ms / Sample")

    print("CPU:")
    print(f"AutoLog: {al_throughput_cpu:.4}ms / Sample")
    print(f"CLDTLog: {cldtlog_throughput_cpu:.4}ms / Sample")

    plot_throughput({
        "AutoLog": {
            "CPU": al_throughput_cpu,
            "GPU": al_throughput_cuda
        },
        "CLDTLog": {
            "CPU": cldtlog_throughput_cpu,
            "GPU": cldtlog_throughput_cuda
        }
    }, outname=r".\\figures\\model_throughput_comparison.png")
