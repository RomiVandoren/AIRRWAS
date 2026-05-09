import warnings

warnings.filterwarnings("ignore")

import logging
import multiprocessing
import os
import time

from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from raptcr.neighborhood import ConvergenceAnalysis, Fisher
from raptcr.hashing import TCRDistEmbedder

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger().handlers[0].flush()


# Configuration for multiprocessing
N_PROCESSES = 12
N_ITERATIONS = 10
BASE_RANDOM_SEED = 72

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input/output directories
DATA_DIR = BASE_DIR / "2_processed_data" / "3_discovery_convergence"
OUTPUT_DIR = BASE_DIR / "2_processed_data" / "11_negative_networks" / "0_shuffled_PB_PL"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_vgene(dataframe, group_col, pos_list, unique_vgene):
    try:
        # process_id = os.getpid()
        # logging.info(f"Process {process_id} is processing V-gene: {unique_vgene}")
        # start_time = time.time()

        df = dataframe.query(f"v_call == '{unique_vgene}'").copy()
        tcrdist = TCRDistEmbedder(full_tcr=False).fit()
        fisher = Fisher(group_column=group_col, positive_groups=pos_list.copy())
        cva = ConvergenceAnalysis(
            convergence_metric=fisher, tcr_embedder=tcrdist, verbose=False
        )
        result = cva.fit_transform(df)

        # logging.info(
        #     f"Processed {unique_vgene} in {time.time() - start_time:.2f}s by process {process_id}"
        # )
        return result
    except Exception as e:
        logging.error(f"Error processing {unique_vgene}: {e}")
        return pd.DataFrame()


def get_per_gene_convergence(dataframe, group_col, pos_list):
    vgenes = dataframe["v_call"].unique()
    func = partial(process_vgene, dataframe, group_col, pos_list)

    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        results = []
        for res in tqdm(
            pool.imap_unordered(func, vgenes),
            total=len(vgenes),
            desc="Processing V-genes",
        ):
            results.append(res)

    return pd.concat(results, ignore_index=True)


def main():
    # === Load data ===
    datasets_info = [
        {
            "name": "alpha",
            "data_path": DATA_DIR / "TRA_dataframe.csv",
            "group_col": "patient_id",
            "chain": "alpha",
            "duplicates": [
                "junction",
                "junction_aa",
                "v_call",
                "j_call",
                "repertoire_id",
            ],
        },
        {
            "name": "beta",
            "data_path": DATA_DIR / "TRB_dataframe.csv",
            "group_col": "patient_id",
            "chain": "beta",
            "duplicates": [
                "junction",
                "junction_aa",
                "v_call",
                "j_call",
                "repertoire_id",
            ],
        },
    ]

    microbiome_path = DATA_DIR / "genus_count_matrix_th1.csv"

    genus_names = [
        "Prevotella",
        "Peptoniphilus",
    ]

    # -------------------------------------------------------------------------
    # Load microbiome data
    # -------------------------------------------------------------------------

    logging.info("Loading microbiome data...")

    microbiome_df = pd.read_csv(
        microbiome_path,
        index_col=0,
    )

    # -------------------------------------------------------------------------
    # Load TCR datasets
    # -------------------------------------------------------------------------

    logging.info("Loading TCR datasets...")

    for info in datasets_info:

        logging.info(f"Loading {info['name']} dataset")

        df = pd.read_csv(
            info["data_path"],
            # index_col=0,
        )

        info["data"] = df.drop_duplicates(subset=info["duplicates"])

    # === Loop over iterations ===
    for iteration in range(1, N_ITERATIONS + 1):

        seed = BASE_RANDOM_SEED + iteration
        logging.info(f"Starting iteration {iteration} (seed={seed})")

        # Shuffle microbiome labels
        shuffled_microbiome = microbiome_df.sample(
            frac=1,
            axis=0,
            random_state=seed,
        ).reset_index(drop=True)
        shuffled_microbiome.index = microbiome_df.index  # keep patient IDs aligned

        # ---------------------------------------------------------------------
        # Run convergence analysis
        # ---------------------------------------------------------------------

        for info in datasets_info:

            for genus in genus_names:

                if genus not in shuffled_microbiome.columns:
                    logging.warning(f"{genus} not found in microbiome data")
                    continue

                patients = shuffled_microbiome.loc[
                    shuffled_microbiome[genus] > 0
                ].index.tolist()

                logging.info(
                    f"Genus: {genus} | "
                    f"Chain: {info['chain']} | "
                    f"Patients: {len(patients)}"
                )

                result = get_per_gene_convergence(
                    dataframe=info["data"],
                    group_col=info["group_col"],
                    pos_list=patients,
                )

                output_file = (
                    OUTPUT_DIR / f"control_{genus}_{info['chain']}_iteration{seed}.csv"
                )

                result.to_csv(output_file, index=False)

                logging.info(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()  # automatically run 10 shuffled iterations
