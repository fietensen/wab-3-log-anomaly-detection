"""
Handles the parsing of Logs as described in the AutoLog paper
"""

from pathlib import Path
from typing import Callable, Iterator
import parse_util as pu
import datetime
import pandas as pd
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


def process_bgl_line(line: str) -> tuple[str]:
    proc = pu.process_line(
        line,
        BGL_LINE_RE,
        [0, 3, 4, 9]
    )

    if not proc:
        return (None, None, None, None)

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
    ])


def process_hdfs_line(line: str) -> tuple[str]:
    proc = pu.process_line(
        line,
        HDFS_LINE_RE
    )

    if not proc:
        return (None, None, None, None)

    # parse time
    time_parsed = datetime.datetime.strptime(
        proc[0], "%y%m%d %H%M%S"
    )

    le = proc[1] + "-" + proc[2]
    cont_parsed = pu.format_text(proc[3])

    return tuple([
        False,
        le,
        time_parsed,
        cont_parsed
    ])


def process_logfile(path: Path, line_processor: Callable, print_progress: bool = False, print_interval: int = 10_000) -> pd.DataFrame:

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
                           "anomalous", "log_entity", "timestamp", "message"]))
            batch.clear()

    # Final batch
    if batch:
        df_list.append(pd.DataFrame(batch, columns=[
                       "anomalous", "log_entity", "timestamp", "message"]))

    df = pd.concat(df_list, ignore_index=True).dropna()
    df["log_entity"] = df["log_entity"].astype(str)

    return df


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
