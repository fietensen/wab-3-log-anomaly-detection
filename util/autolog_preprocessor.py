"""
Handles the parsing of Logs as described in the AutoLog paper
"""

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, OrderedDict
from typing import Callable, Iterator
from pathlib import Path

import multiprocessing as mp
import util.parse_util as pu
import pandas as pd
import numpy as np
import datetime
import os
import re

# Parses line from BG/L Logfile
BGL_LINE_RE = re.compile(
    r"^([^\s]+)\s"   # 0 - Label (anomalous if not "-")
    + r"([^\s]+)\s"  # 1 - Timestamp
    + r"([^\s]+)\s"  # 2 - Date
    + r"([^\s]+)\s"  # 3 - Node
    + r"([^\s]+)\s"  # 4 - Time
    + r"([^\s]+)\s"  # 5 - NodeRepeat
    + r"([^\s]+)\s"  # 6 - Type
    + r"([^\s]+)\s"  # 7 - Component
    + r"([^\s]+)\s"  # 8 - Level
    + r"(.*)$"       # 9 - Content
)

# Parses line from HDFS Logfile
HDFS_LINE_RE = re.compile(
    r"^(\d{6}\s\d{6})\s"  # Timestamp
    + r"(\d+)\s"          # PID
    + r"\w+\s([^:]+):\s"  # Component
    + r"(.*)$"            # Log Message
)

# Matches first blk entry in logfile
HDFS_BLK_RE = re.compile(r"^.*\s(blk_-?\d+)\s.*$")

def _process_bgl_line(line: str) -> tuple[str]:
    proc = pu.process_line(
        line,
        BGL_LINE_RE,
        [0, 3, 4, 9]
    )

    if not proc:
        return (None, None, None, None, None)

    time_parsed = datetime.datetime.strptime(
        proc[2], "%Y-%m-%d-%H.%M.%S.%f"
    )

    cont_parsed = pu.format_text(proc[3])

    # Some edge cases apparently:
    le = proc[1]
    if le == "-":
        le = "UNSPECIFIED"
    elif le == "UNKNOWN_LOCATION":
        le = "UNKNOWNLOCATION"
    elif le.startswith("R"):
        le = le.split("-")[0]
    else:
        # le is "NULL"
        pass

    return tuple([
        (proc[0] != "-"),         # Anomalous
        le,                       # Log Entity
        time_parsed.timestamp(),  # Time
        cont_parsed,              # Content
        None
    ])


def _process_hdfs_line(line: str) -> tuple[str]:
    proc = pu.process_line(
        line,
        HDFS_LINE_RE
    )

    if not proc:
        return (None, None, None, None, None)

    # parse time
    time_parsed = datetime.datetime.strptime(
        proc[0], "%y%m%d %H%M%S"
    )

    le = proc[2]
    cont_parsed = pu.format_text(proc[3])
    blk_id = HDFS_BLK_RE.match(proc[3])
    if not blk_id:
        return (None, None, None, None, None)

    return tuple([
        False,
        le,
        time_parsed.timestamp(),
        cont_parsed,
        blk_id.groups()[0]
    ])


def _process_loglines_batch(lines: list, line_processor: Callable) -> list:
    return [line_processor(line) for line in lines]


def _process_logfile(path: Path, line_processor: Callable, print_progress: bool = False, print_interval: int = 10_000) -> pd.DataFrame:
    def log_generator(file_path: Path) -> Iterator[list]:
        with open(file_path, "r", encoding="utf8") as fp:
            batch = []
            for i, line in enumerate(fp):
                if i % print_interval == 0 and print_progress:
                    print(f"[INFO] Reading Log Lines: {i:,}\r", end="")

                batch.append(line)

                if len(batch) >= chunk_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

        if print_progress:
            print("\n[INFO] Read Log Lines. Processing.. This may take a while.")

    chunk_size = 10_000
    df_list = []

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(
            _process_loglines_batch, (batch, line_processor)) for batch in log_generator(path)]
        processed_batches = [r.get() for r in results]

    for processed_batch in processed_batches:
        df_list.append(pd.DataFrame(processed_batch, columns=[
                       "anomalous", "log_entity", "timestamp", "message", "ext"]))

    df = pd.concat(df_list, ignore_index=True).dropna(how="all")
    df.index.name = "index"
    df["log_entity"] = df["log_entity"].astype(str)

    return df

def _set_hdfs_labels(df: pd.DataFrame, label_file: Path) -> pd.DataFrame:
    an = pd.read_csv(label_file).set_index("BlockId").to_dict()["Label"]
    df["anomalous"] = df["ext"].map(lambda blk_id: an[blk_id] == "Anomaly")

def prepare_hdfs(hdfs_dataset_path: Path) -> None:
    hdfs_log_file = hdfs_dataset_path / "HDFS.log"
    hdfs_label_file = hdfs_dataset_path / "preprocessed" / "anomaly_label.csv"
    hdfs_log_parsed = hdfs_dataset_path / "HDFSL.prep.csv"

    parsed = None

    if os.path.isfile(hdfs_log_parsed):
        print("[*] Loading cached preprocessed file from " + str(hdfs_log_parsed))
        parsed = pd.read_csv(hdfs_log_parsed)
        parsed.index.name = "index"

    else:
        parsed = _process_logfile(
            hdfs_log_file,
            line_processor=_process_hdfs_line,
            print_progress=True
        )
        print("[INFO] Setting labels")
        _set_hdfs_labels(parsed, hdfs_label_file)
        print("[INFO] Caching progress to " + str(hdfs_log_parsed))
        del parsed["ext"]
        parsed.to_csv(hdfs_log_parsed, index=False)

    return parsed


def prepare_bgl(bgl_dataset_path: Path) -> pd.DataFrame:
    bgl_log_file = bgl_dataset_path / "BGL.log"
    bgl_log_parsed = bgl_dataset_path / "BGL.prep.csv"

    parsed = None

    if os.path.isfile(bgl_log_parsed):
        print("[INFO] Loading cached preprocessed file from " + str(bgl_log_parsed))
        parsed = pd.read_csv(bgl_log_parsed)
        parsed.index.name = "index"
        parsed["log_entity"] = parsed["log_entity"].replace(np.nan, "NULL")

    else:
        parsed = _process_logfile(
            bgl_log_file,
            line_processor=_process_bgl_line,
            print_progress=True
        )
        print("[INFO] Caching progress to " + str(bgl_log_parsed))
        del parsed["ext"]
        parsed.to_csv(bgl_log_parsed, index=False)

    return parsed

def mk_chunks_vectorizers(df: pd.DataFrame, time_interval: int = 10) -> tuple[dict[str, CountVectorizer], pd.DataFrame, pd.DataFrame]:# -> tuple[dict[str, dict[int, int]], dict[str, CountVectorizer], dict[str, pd.DataFrame]]:
    ts_start = df["timestamp"].min()

    df["tblk"] = ((df["timestamp"]-ts_start)//time_interval).astype(int)
    tblk_groups = df.groupby("tblk")
    df["anomalous"] = tblk_groups["anomalous"].transform("any")

    le_vectorizers = defaultdict(lambda:CountVectorizer(token_pattern=r"(?u)\b\w\w*\b"))
    norm_df = df[df["anomalous"] == False]
    anom_df = df[df["anomalous"] == True]

    for le_name, gdf in df.groupby("log_entity"):
        le_vectorizers[le_name].fit([" ".join(gdf["message"].values)])

    return le_vectorizers, norm_df, anom_df

def calc_term_entropies(vectorizer: CountVectorizer, le_messages: pd.Series, M: int) -> np.array:
    sp_mat = vectorizer.transform(le_messages)

    # e_t = 1 + 1/log2(M) * (\sum_{j}^{M} p_t_j*log2(p_t_j))
    abs_sum = np.asarray(sp_mat.sum(axis=0)).flatten()
    abs_sum[abs_sum == 0] = 1 # avoid division by zero
    norm_sp_mat = sp_mat.multiply(1 / abs_sum)
    p_log_p = norm_sp_mat.copy()
    p_log_p.data = p_log_p.data * np.log2(p_log_p.data)
    entropy_t = np.asarray(p_log_p.sum(axis=0)).flatten()
    H_t = 1 + (entropy_t / np.log2(M))
    
    return H_t

def calc_chunk_scores(vectorizer: CountVectorizer, entropies: np.array, chunks: list[str]) -> np.array:
    sp_mat = vectorizer.transform(chunks)
    sp_mat.data = np.log2(1 + sp_mat.data)
    sp_mat = sp_mat.multiply(entropies).power(2).sum(axis=1)

    chunk_scores = np.array(np.sqrt(sp_mat)).flatten()

    return chunk_scores


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")