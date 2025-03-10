from typing import Callable, Iterator
from pathlib import Path
from transformers import BertTokenizer

import multiprocessing as mp
import util.parse_util as pu
import numpy as np
import pandas as pd
import h5py
import random
import datetime
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

def _process_bgl_line(line: str) -> tuple[str]:
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

    cont_parsed = pu.format_text(proc[3], include_digits=False, to_lowercase=False)

    return tuple([
        (proc[0] != "-"),
        time_parsed.timestamp(),
        cont_parsed,
        None
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
                       "anomalous", "timestamp", "message", "ext"]))

    df = pd.concat(df_list, ignore_index=True).dropna(how="all")
    df.index.name = "index"

    return df


def prepare_bgl(bgl_dataset_path: Path, window_size: int = 5, window_step: int = 2) -> tuple[list[str], list[bool]]:
    bgl_log_file = bgl_dataset_path / "BGL.log"
    bgl_log_parsed = bgl_dataset_path / "BGL.prep.cldt.csv"

    parsed = None

    if os.path.isfile(bgl_log_parsed):
        print("[INFO] Loading cached preprocessed file from " + str(bgl_log_parsed))
        parsed = pd.read_csv(bgl_log_parsed)
        parsed.index.name = "index"

    else:
        parsed = _process_logfile(
            bgl_log_file,
            line_processor=_process_bgl_line,
            print_progress=True
        )
        print("[INFO] Caching progress to " + str(bgl_log_parsed))
        del parsed["ext"]
        parsed.to_csv(bgl_log_parsed, index=False)

    df = parsed.dropna()
    messages = []
    labels = []

    for i in df[0::window_step].index:
        window = df[i : i+window_size]
        messages.append(" ".join(window["message"].values))
        labels.append(window["anomalous"].any())

    mlpairs = list(zip(messages, labels))
    random.shuffle(mlpairs)
    messages, labels = list(zip(*mlpairs))

    return messages, labels


def load_tokenizer(path: Path):
    return BertTokenizer.from_pretrained(path)


def create_h5_file(output_file: Path):
    with h5py.File(output_file, "w") as f:
        f.create_dataset("/data/normative/input_ids", (0, 50), maxshape=(None, 50), dtype="uint16", chunks=True)
        f.create_dataset("/data/normative/attention_mask", (0, 50), maxshape=(None, 50), dtype="uint8", chunks=True)
        
        f.create_dataset("/data/anomalous/input_ids", (0, 50), maxshape=(None, 50), dtype="uint16", chunks=True)
        f.create_dataset("/data/anomalous/attention_mask", (0, 50), maxshape=(None, 50), dtype="uint8", chunks=True)


def append_to_h5(input_ids: np.array, attention_mask: np.array, labels: np.array, output_file: Path):
    with h5py.File(output_file, "a") as f:
        norm_input_ids = input_ids[~labels]
        norm_attention_mask = attention_mask[~labels]

        anom_input_ids = input_ids[labels]
        anom_attention_mask = attention_mask[labels]

        new_size_norm = f["/data/normative/input_ids"].shape[0] + norm_input_ids.shape[0]
        new_size_anom = f["/data/anomalous/input_ids"].shape[0] + anom_input_ids.shape[0]

        if (~labels).any():
            f["/data/normative/input_ids"].resize((new_size_norm, 50))
            f["/data/normative/attention_mask"].resize((new_size_norm, 50))

            f["/data/normative/input_ids"][-norm_input_ids.shape[0]:] = norm_input_ids
            f["/data/normative/attention_mask"][-norm_attention_mask.shape[0]:] = norm_attention_mask


        if (labels).any():
            f["/data/anomalous/input_ids"].resize((new_size_anom, 50))
            f["/data/anomalous/attention_mask"].resize((new_size_anom, 50))

            f["/data/anomalous/input_ids"][-anom_input_ids.shape[0]:] = anom_input_ids
            f["/data/anomalous/attention_mask"][-anom_attention_mask.shape[0]:] = anom_attention_mask


def create_tokenize_message_dataset(messages: list[str], labels: list[bool], h5_file_path: Path, tokenizer_path: Path, batch_size: int = 10_000):
    create_h5_file(h5_file_path)
    tokenizer = load_tokenizer(tokenizer_path)

    for i in range(0, len(messages), batch_size):
        chunk_messages = messages[i : i + batch_size]
        chunk_labels = labels[i : i + batch_size]

        chunk_tokenized = tokenizer(chunk_messages, return_tensors="np", truncation=True, max_length=50, padding="max_length")
        chunk_input_ids = chunk_tokenized["input_ids"]
        chunk_attention_mask = chunk_tokenized["attention_mask"]
        append_to_h5(chunk_input_ids, chunk_attention_mask, np.array(chunk_labels, dtype=bool), h5_file_path)


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
