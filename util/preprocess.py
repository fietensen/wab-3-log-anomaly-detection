import util.autolog_preprocessor as al_prep 
import util.cldtlog_preprocessor as cldt_prep
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

def preprocess_autolog(raw_df: pd.DataFrame, time_interval: int = 10) -> pd.DataFrame:
    print("[INFO] Loaded DataFrame")
    
    vectorizers, normdf, anomdf = al_prep.mk_chunks_vectorizers(raw_df, time_interval)
    print("[INFO] Loaded and fitted vectorizers")

    data = []

    # Normative
    le_entropies = {}
    for le, legrp in raw_df.groupby("log_entity"):
        tblks, chunks = zip(*[
            (tblk, " ".join(gdf["message"].values))
                for tblk, gdf in legrp.groupby("tblk")
        ])

        le_term_entropies = al_prep.calc_term_entropies(vectorizers[le], chunks, len(chunks))
        le_entropies[le] = le_term_entropies

    for le, legrp in normdf.groupby("log_entity"):
        tblks, chunks = zip(*[
            (tblk, " ".join(gdf["message"].values))
                for tblk, gdf in legrp.groupby("tblk")
        ])

        le_term_entropies = le_entropies[le]
        le_chunk_scores = al_prep.calc_chunk_scores(
            vectorizers[le],
            le_term_entropies,
            chunks
        )

        for tblk, chunk_score in zip(tblks, le_chunk_scores):
            data.append([le, tblk, chunk_score, False])

    # Anomalous
    for le, legrp in anomdf.groupby("log_entity"):
        tblks, chunks = zip(*[
            (tblk, " ".join(gdf["message"].values))
                for tblk, gdf in legrp.groupby("tblk")
        ])

        le_term_entropies = le_entropies[le]
        le_chunk_scores = al_prep.calc_chunk_scores(vectorizers[le], le_term_entropies, chunks)

        for tblk, chunk_score in zip(tblks, le_chunk_scores):
            data.append([le, tblk, chunk_score, True])
    
    sdf = pd.DataFrame(data, columns=["log_entity", "tblk", "score", "anomalous"])
    df_pivot = sdf.pivot_table(index='tblk', columns='log_entity', values='score', aggfunc='first')
    df_pivot.fillna(0, inplace=True)
    anomalous = sdf.groupby('tblk')['anomalous'].max()

    df_pivot = df_pivot.merge(anomalous, left_index=True, right_index=True, how='left')
    df_pivot.reset_index(inplace=True)

    return df_pivot


def preprocess_cldtlog(messages: list[str], labels: list[bool], tokenizer_model: Path, h5_path: Path, batch_size: int = 10_000):
    cldt_prep.create_tokenize_message_dataset(messages, labels, h5_path, tokenizer_model, batch_size)


def main():
    bglp = Path(r".\datasets\BGL")
    bglprpal = bglp / "preprocessed.al.csv"
    bglprpcldt = bglp / "preprocessed.cldt.h5"

    hdfsp = Path(r".\datasets\HDFS_v1")
    hdfsprpal = hdfsp / "preprocessed.al.csv"
    hdfsprpcldt = hdfsp / "preprocessed.cldt.h5"

    tokenizer_path = Path(r".\bert_base_uncased_hf_tokenizer")

    # AutoLog Preprocessing
    print("[INFO] Preparing and Preprocessing BG/L Dataset for AutoLog")
    al_bgl_proc = preprocess_autolog(al_prep.prepare_bgl(bglp), time_interval=300)
    print("[INFO] Storing preprocessed Dataframe at " + str(bglprpal))
    al_bgl_proc.to_csv(bglprpal, index=False)


    print("[INFO] Preparing and Preprocessing HDFS Dataset for AutoLog")
    al_hdfs_proc = preprocess_autolog(al_prep.prepare_hdfs(hdfsp), time_interval=10)
    print("[INFO] Storing preprocessed Dataframe at " + str(hdfsprpal))
    al_hdfs_proc.to_csv(hdfsprpal, index=False)

    # CLDTLog Preprocessing
    print("[INFO] Preparing and Preprocessing BG/L Dataset for CLDTLog")
    preprocess_cldtlog(*cldt_prep.prepare_bgl(bglp), tokenizer_path, bglprpcldt)
    print("[INFO] Storing preprocessed Dataset at " + str(bglprpcldt))

    #print("[INFO] Preparing and Preprocessing BG/L Dataset for CLDTLog")
    #print("[INFO] Storing preprocessed Dataframe at " + str(bglprpcldt))



if __name__ == '__main__':
    main()