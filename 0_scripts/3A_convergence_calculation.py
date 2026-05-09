import logging
import multiprocessing
import os
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from raptcr.neighborhood import ConvergenceAnalysis, Fisher
from raptcr.hashing import TCRDistEmbedder

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger().handlers[0].flush()


# Per gene processing function
def process_vgene(dataframe, group_col, pos_list, unique_vgene):
    try:
        process_id = os.getpid()
        logging.info(f"Process {process_id} is processing V-gene: {unique_vgene}")

        start_time = time.time()
        df = dataframe.query(f"v_call == '{unique_vgene}'").copy()
        tcrdist = TCRDistEmbedder(full_tcr=False).fit()
        fisher = Fisher(group_column=group_col, positive_groups=pos_list.copy())
        cva = ConvergenceAnalysis(
            convergence_metric=fisher,
            tcr_embedder=tcrdist,
            verbose=False,
        )
        result = cva.fit_transform(df)
        end_time = time.time()
        logging.info(
            f"Processed {unique_vgene} in {end_time - start_time:.2f}s by process {process_id}"
        )
        return result
    except Exception as e:
        logging.error(f"Error processing {unique_vgene}: {e}")
        return pd.DataFrame()


def get_per_gene_convergence(dataframe, group_col, pos_list):
    vgenes = dataframe["v_call"].unique()
    func = partial(process_vgene, dataframe, group_col, pos_list)

    with multiprocessing.Pool(processes=12) as pool:
        results = [
            res
            for res in tqdm(
                pool.imap_unordered(func, vgenes),
                total=len(vgenes),
                desc="Processing V-genes",
            )
        ]
    return pd.concat(results, ignore_index=True)


# Calculate convergence
def main():
    # Directories
    base_dir = Path("..").resolve()
    disc_conv_dir = base_dir / "2_processed_data" / "3_discovery_convergence"
    ibd_conv_dir = base_dir / "2_processed_data" / "4_Brandt_convergence"
    crc_conv_dir = base_dir / "2_processed_data" / "5_Pu_convergence"

    # Datasets
    datasets_info = [
        {
            "data_path": disc_conv_dir / "TRA_dataframe.csv",
            "micro_path": disc_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "patient_id",
            "chain": "alpha",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "repertoire_id"],
            "save_dir": disc_conv_dir / "genus_results",
        },
        {
            "data_path": disc_conv_dir / "TRB_dataframe.csv",
            "micro_path": disc_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "patient_id",
            "chain": "beta",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "repertoire_id"],
            "save_dir": disc_conv_dir / "genus_results",
        },
        {
            "data_path": ibd_conv_dir / "TRA_dataframe.csv",
            "micro_path": ibd_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "patient_id",
            "chain": "alpha",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "patient_id"],
            "save_dir": ibd_conv_dir / "genus_results",
        },
        {
            "data_path": ibd_conv_dir / "TRB_dataframe.csv",
            "micro_path": ibd_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "patient_id",
            "chain": "beta",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "patient_id"],
            "save_dir": ibd_conv_dir / "genus_results",
        },
        {
            "data_path": crc_conv_dir / "TRA_dataframe.csv",
            "micro_path": crc_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "repertoire_id",
            "chain": "alpha",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "repertoire_id"],
            "save_dir": crc_conv_dir / "genus_results",
        },
        {
            "data_path": crc_conv_dir / "TRB_dataframe.csv",
            "micro_path": crc_conv_dir / "genus_count_matrix_matched.csv",
            "group_col": "repertoire_id",
            "chain": "beta",
            "dupl": ["junction", "junction_aa", "v_call", "j_call", "repertoire_id"],
            "save_dir": crc_conv_dir / "genus_results",
        },
    ]

    for info in datasets_info:
        info["save_dir"].mkdir(exist_ok=True, parents=True)
        info["data"] = pd.read_csv(info["data_path"]).drop_duplicates(
            subset=info["dupl"]
        )
        info["micro"] = pd.read_csv(info["micro_path"], index_col=0)

    # Get overlapping genera across all datasets
    overlap_list = set.intersection(
        *(set(info["micro"].columns) for info in datasets_info)
    )
    logging.info(f"Number of overlapping genera: {len(overlap_list)}")

    # Loop over genera and datasets
    for genus in overlap_list:
        for info in datasets_info:
            patients = info["micro"].loc[info["micro"][genus] > 0].index.tolist()
            logging.info(
                f"Processing genus {genus} for {info['chain']} chain, {len(patients)} patients"
            )

            result = get_per_gene_convergence(info["data"], info["group_col"], patients)
            result.to_csv(
                info["save_dir"] / f"{genus}_{info['chain']}_results.csv", index=False
            )


if __name__ == "__main__":
    main()
