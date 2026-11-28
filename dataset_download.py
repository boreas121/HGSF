# -*- coding: utf-8 -*-
"""
dataset_download.py -- Download HEST v1.1.0 from HuggingFace MahmoodLab/hest
kernel: https://github.com/mahmoodlab/HEST
Paths anonymized: everything is downloaded under /data/hest1k_datasets/
"""
import os
import datasets
import pandas as pd
from huggingface_hub import login

DATA_ROOT = "/data/hest1k_datasets"


def main():
    # ---- HuggingFace login ----
    login(token="")  # token = Your_HuggingFace_Token

    # ---- load metadata ----
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

    def download(data_path, ids_to_query):
        os.makedirs(data_path, exist_ok=True)
        list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
        return datasets.load_dataset(
            'MahmoodLab/hest',
            cache_dir=data_path,
            patterns=list_patterns,
        )

    # ============ Dataset - Human HER2ST ============
    ids_to_query = meta_df.loc[
        meta_df["dataset_title"].str.startswith("Spatial deconvolution of"),
        "id"].tolist()
    download(f"{DATA_ROOT}/her2st", ids_to_query)
    # Test slide in the paper: SPA148

    # ============ Dataset - cSCC ============
    download(f"{DATA_ROOT}/cSCC", ["NCBI" + str(i) for i in range(759, 771)])
    # Test slide in the paper: NCBI759

    # ============ Dataset - Human Prostate Cancer (PRAD) ============
    download(f"{DATA_ROOT}/PRAD", ["MEND" + str(i) for i in range(139, 163)])  # ID: MEND139~MEND162
    # Test slide in the paper: MEND145

    # ============ Dataset - DLPFC ============
    download(f"{DATA_ROOT}/DLPFC", ["MISC" + str(i) for i in range(1, 13)])    # ID: MISC1~MISC12
    # Test slide in the paper: MISC2


if __name__ == '__main__':
    main()
