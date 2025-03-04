from util.autolog_preprocessor import mk_chunks_vectorizers, prepare_hdfs, prepare_bgl, calc_term_entropies, calc_chunk_scores
from pathlib import Path

import pandas as pd

def preprocess_autolog(raw_df: pd.DataFrame, time_interval: int = 10) -> pd.DataFrame:
    print("[INFO] Loaded DataFrame")
    
    vectorizers, normdf, anomdf = mk_chunks_vectorizers(raw_df, time_interval)
    print("[INFO] Loaded and fitted vectorizers")

    data = []

    # Normative
    le_entropies = {}
    for le, legrp in raw_df.groupby("log_entity"):
        tblks, chunks = zip(*[
            (tblk, " ".join(gdf["message"].values))
                for tblk, gdf in legrp.groupby("tblk")
        ])

        le_term_entropies = calc_term_entropies(vectorizers[le], chunks, len(chunks))
        le_entropies[le] = le_term_entropies

    for le, legrp in normdf.groupby("log_entity"):
        tblks, chunks = zip(*[
            (tblk, " ".join(gdf["message"].values))
                for tblk, gdf in legrp.groupby("tblk")
        ])

        le_term_entropies = le_entropies[le]
        le_chunk_scores = calc_chunk_scores(
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
        le_chunk_scores = calc_chunk_scores(vectorizers[le], le_term_entropies, chunks)

        for tblk, chunk_score in zip(tblks, le_chunk_scores):
            data.append([le, tblk, chunk_score, True])
    
    sdf = pd.DataFrame(data, columns=["log_entity", "tblk", "score", "anomalous"])
    df_pivot = sdf.pivot_table(index='tblk', columns='log_entity', values='score', aggfunc='first')
    df_pivot.fillna(0, inplace=True)
    anomalous = sdf.groupby('tblk')['anomalous'].max()

    df_pivot = df_pivot.merge(anomalous, left_index=True, right_index=True, how='left')
    df_pivot.reset_index(inplace=True)

    return df_pivot

def main():
    bglp = Path(r".\datasets\BGL")
    bglprp = bglp / "preprocessed.al.csv"
    
    hdfsp = Path(r".\datasets\HDFS_v1")
    hdfsprp = hdfsp / "preprocessed.al.csv"

    print("[INFO] Preparing and Preprocessing BG/L Dataset for AutoLog")
    al_bgl_proc = preprocess_autolog(prepare_bgl(bglp), time_interval=300)
    print("[INFO] Storing preprocessed Dataframe at " + str(bglprp))
    al_bgl_proc.to_csv(bglprp, index=False)


    print("[INFO] Preparing and Preprocessing HDFS Dataset for AutoLog")
    al_hdfs_proc = preprocess_autolog(prepare_hdfs(hdfsp), time_interval=10)
    print("[INFO] Storing preprocessed Dataframe at " + str(hdfsprp))
    al_hdfs_proc.to_csv(hdfsprp, index=False)

if __name__ == '__main__':
    main()