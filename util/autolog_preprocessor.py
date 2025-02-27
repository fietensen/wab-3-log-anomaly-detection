"""
Handles the parsing of Logs as described in the AutoLog paper
"""

from pathlib import Path
from typing import Callable, Iterator
import multiprocessing as mp
import parse_util as pu
import datetime
import pandas as pd
import re
import os

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
        (proc[0] != "-"),  # Anomalous
        le,                # Log Entity
        time_parsed,       # Time
        cont_parsed        # Content
    ]), None


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

    le = proc[1] + "-" + proc[2]
    cont_parsed = pu.format_text(proc[3])
    blk_id = HDFS_BLK_RE.match(proc[3])
    if not blk_id:
        return (None, None, None, None, None)

    return tuple([
        False,
        le,
        time_parsed,
        cont_parsed,
        blk_id.groups()[0]
    ])


"""
def _process_logfile(path: Path, line_processor: Callable, print_progress: bool = False, print_interval: int = 10_000) -> pd.DataFrame:
    def process_batch(lines: list) -> list:
        return [line_processor(line) for line in lines]

    def log_generator(file_path: Path) -> Iterator[tuple]:
        with open(file_path, "r", encoding="utf8") as fp:
            for i, line in enumerate(fp):
                if i % print_interval == 0 and print_progress:
                    print(f"[INFO] Processed Log Lines: {i:,}\r", end="")
                yield line_processor(line)
        if print_progress:
            print("\n[INFO] Processed Logfile")

    chunk_size = 10000
    df_list = []

    batch = []
    for i, parsed_line in enumerate(log_generator(path)):
        batch.append(parsed_line)

        if len(batch) >= chunk_size:
            df_list.append(pd.DataFrame(batch, columns=[
                           "anomalous", "log_entity", "timestamp", "message", "ext"]))
            batch.clear()

    # Final batch
    if batch:
        df_list.append(pd.DataFrame(batch, columns=[
                       "anomalous", "log_entity", "timestamp", "message", "ext"]))

    df = pd.concat(df_list, ignore_index=True).dropna()
    df["log_entity"] = df["log_entity"].astype(str)

    return df
"""


def _process_batch(lines: list, line_processor: Callable) -> list:
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
            _process_batch, (batch, line_processor)) for batch in log_generator(path)]
        processed_batches = [r.get() for r in results]

    for processed_batch in processed_batches:
        df_list.append(pd.DataFrame(processed_batch, columns=[
                       "anomalous", "log_entity", "timestamp", "message", "ext"]))

    df = pd.concat(df_list, ignore_index=True).dropna()
    df["log_entity"] = df["log_entity"].astype(str)

    return df


def _set_hdfs_labels(df: pd.DataFrame, label_file: Path) -> None:
    an_labels = pd.read_csv(label_file)
    df["anomalous"] = df.map(
        lambda idx: an_labels[
            an_labels["BlockId"] == df.iloc[idx]["ext"]
        ].values[0] == "Anomaly"
    )


def prepare_hdfs(hdfs_dataset_path: Path) -> None:
    hdfs_log_file = hdfs_dataset_path / "HDFS.log"
    hdfs_label_file = hdfs_dataset_path / "preprocessed" / "anomaly_label.csv"
    hdfs_log_parsed = hdfs_dataset_path / "HDFSL.prep.csv"

    parsed = None

    if os.path.isfile(hdfs_log_parsed):
        print("[*] Loading cached preprocessed file from " + hdfs_log_parsed)
        parsed = pd.read_csv(hdfs_log_parsed)

    else:
        parsed = _process_logfile(
            hdfs_log_file,
            line_processor=_process_hdfs_line,
            print_progress=True
        )
        print("[*] Setting labels")
        parsed.to_csv(hdfs_log_parsed)
        _set_hdfs_labels(parsed, hdfs_label_file)
        print("[*] Caching progress to " + hdfs_log_parsed)
        parsed.to_csv(hdfs_log_parsed)

    print(parsed.head())


def prepare_bgl(bgl_dataset_path: Path) -> None:
    bgl_log_file = bgl_dataset_path / "BGL.log"
    bgl_log_parsed = bgl_dataset_path / "BGL.prep.csv"

    parsed = None

    if os.path.isfile(bgl_log_parsed):
        print("[*] Loading cached preprocessed file from " + bgl_log_parsed)
        parsed = pd.read_csv(bgl_log_parsed)

    else:
        parsed = _process_logfile(
            bgl_log_file,
            line_processor=_process_bgl_line,
            print_progress=True
        )
        print("[*] Caching progress to " + bgl_log_parsed)
        parsed.to_csv(bgl_log_parsed)

    print(parsed.head())


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
