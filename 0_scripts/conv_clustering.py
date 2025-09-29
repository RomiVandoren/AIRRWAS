# Imports
import re
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from scipy.stats import rankdata, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, chi2_contingency


# ============================================================================================== #
### Data processing functions ###
# ============================================================================================== #


# Set of standard amino acids
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


#############################
### TCR parsing functions ###
#############################


def _is_aaseq(seq: str) -> bool:
    """Return True if sequence contains only standard amino acids."""
    if not isinstance(seq, str):
        return False
    return all(c in _AMINO_ACIDS for c in seq)


def _is_cdr3(seq: str) -> bool:
    """Return True if sequence is a plausible CDR3 (IMGT rules)."""
    if not isinstance(seq, str):
        return False
    return (
        _is_aaseq(seq)
        and seq.startswith("C")
        and seq[-1] in {"F", "W", "C"}
        and 4 <= len(seq) <= 30
    )


def _select_gene(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Transform gene call column to a consistent format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name containing gene calls.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned gene call formatting.
    """
    df[column] = df[column].apply(
        lambda x: x if pd.isna(x) else x.split(",")[0].split("(")[0]
    )
    df[column] = df[column].apply(
        lambda x: x.split("*")[0] + "*01" if pd.notna(x) else x
    )
    return df


def parse_mixcr(df: pd.DataFrame, rename: bool = True) -> pd.DataFrame:
    """
    Parse MiXCR output and filter for functional CDR3 sequences.

    Parameters
    ----------
    df : pd.DataFrame
        MiXCR output table (raw or pre-renamed).
    rename : bool, default=True
        Whether to rename MiXCR columns to a standard schema.

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered MiXCR DataFrame.
    """
    base_dir = Path(__file__).resolve().parents[1]
    imgt_file = base_dir.parent / "2_processed_data" / "imgt_reference.tsv"
    imgt = pd.read_csv(imgt_file, sep="\t")
    functional = imgt[imgt["fct"].isin({"F", "(F)", "[F]"})]

    if rename:
        df = df.rename(
            columns={
                "cloneId": "clone_id",
                "cloneCount": "duplicate_count",
                "nSeqCDR3": "junction",
                "aaSeqCDR3": "junction_aa",
                "allVHitsWithScore": "v_call",
                "allJHitsWithScore": "j_call",
            }
        )

    # Filter by functional genes
    df = _select_gene(df, "v_call")
    df = df[df["v_call"].isin(functional["imgt_allele_name"])]

    df = _select_gene(df, "j_call")
    df = df[df["j_call"].isin(functional["imgt_allele_name"])]

    # Filter by valid CDR3 sequences
    df = df[df["junction_aa"].apply(_is_cdr3)]

    return df


#####################################
### Get convergent TCRs per genus ###
#####################################


def load_genus_results(
    results_dir: Path,
    convergence_threshold=2,
    pvalue_threshold=0.05,
    file_selection="full_genera",
):
    genus_dfs = []
    for file in results_dir.glob("*.csv"):
        if file_selection == "full_genera":
            match = re.match(r"^(?!X_)(.+?)_(alpha|beta)_results\.csv", file.name)
        if file_selection == "selected_genera":
            match = re.match(r"X_(.+?)_(alpha|beta)_results\.csv", file.name)
        if not match:
            continue
        genus = match.group(1)
        df = pd.read_csv(file, index_col=0)
        df = df[
            (df["convergence"] > convergence_threshold)
            & (df["pvalue"] <= pvalue_threshold)
        ]
        df["genus"] = genus
        print(file.name, len(df))
        genus_dfs.append(df)
    return pd.concat(genus_dfs)


###############################
### Cluster convergent TCRs ###
###############################


def get_clusters(dataframe: pd.DataFrame, savename: str, base_dir: Path = None):
    """
    Load ClusTCR clustering results (or run if needed) and merge with input dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing TCR sequences (must include 'junction_aa').
    savename : str
        Name to use for saving/loading cluster results.
    base_dir : Path, optional
        Base directory for saving/loading cluster results.
        Defaults to manuscript's cluster_convergence folder.

    Returns
    -------
    pd.DataFrame
        Input dataframe with cluster assignments and motif summaries merged in.
    """

    base_dir.mkdir(parents=True, exist_ok=True)

    cluster_pkl = base_dir / f"Clusters_{savename}.pkl"
    cluster_csv = base_dir / f"0_clustered_{savename}.csv"

    # === Run clustering if not already computed ===
    if not cluster_pkl.exists():
        from clustcr.clustering.clustering import Clustering

        print(f"Running ClusTCR on {savename}...")
        res = Clustering(n_cpus=12).fit(data=dataframe["junction_aa"])

        with open(cluster_pkl, "wb") as file:
            pickle.dump(res, file)
    else:
        print(f"Loading existing clustering results for {savename}...")

    # === Load results ===
    with open(cluster_pkl, "rb") as file:
        loaded_res = pickle.load(file)

    # Merge cluster assignments
    data = pd.merge(
        left=dataframe,
        right=loaded_res.clusters_df[["junction_aa", "cluster"]],
        on="junction_aa",
        how="left",
    )
    data["cluster"] = data["cluster"].fillna(-1).astype(int).astype(str)

    # Add motif summaries
    motifs = loaded_res.summary().reset_index()
    motifs.rename(columns={"index": "cluster_nr"}, inplace=True)
    motifs["cluster_nr"] = motifs["cluster_nr"].astype(str)

    data = pd.merge(
        data, motifs, left_on="cluster", right_on="cluster_nr", how="left"
    ).drop(columns="cluster_nr")

    data["genus"] = savename

    # Save merged dataframe
    data.to_csv(cluster_csv, index=False)

    return data


###################################
### Explore convergent clusters ###
###################################


def analyze_clusters(
    genera,
    cluster_dir: Path,
    output_dir: Path,
    convergence_threshold: float = 2,
    pvalue_threshold: float = 0.05,
    top_percentile: float = 10,
):
    """
    Analyze convergent TCR clusters across genera and datasets.

    Parameters
    ----------
    genera : list of str
        List of genera names to analyze.
    cluster_dir : Path
        Directory containing clustered CSVs per genus.
    output_dir : Path
        Directory where results will be saved.
    convergence_threshold : float, optional
        Minimum convergence score to keep (default=2).
    pvalue_threshold : float, optional
        Maximum p-value threshold for significance (default=0.05).
    top_percentile : float, optional
        Percentile for defining "top clusters" (default=10).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    overview_data = []
    raw_pvals = []
    red_only_dfs = []
    all_clustered_dfs = []
    summary_frac_data = []
    all_ranks_data = []
    grouped_data_list = []

    for genus in genera:
        file_path = cluster_dir / f"0_clustered_{genus}.csv"
        if not file_path.exists():
            print(f"Missing file for {genus}, skipping.")
            continue

        df = pd.read_csv(file_path)

        # --- Group clusters ---
        df_grouped = (
            df[df["cluster"] != -1]
            .groupby("cluster")
            .agg(
                mean_convergence=("convergence", "mean"),
                unique_patients=("patient_id", "nunique"),
                unique_datasets=("dataset", "nunique"),
                cluster_size=("junction_aa", "nunique"),
            )
            .reset_index()
        )

        # --- Mark shared clusters (red) ---
        def mark_shared(dsets, genus_name):
            if dsets == 3:
                return "red"
            if genus_name == "Peptoniphilus" and dsets == 2:
                return "red"
            return "lightgrey"

        df_grouped["color_sharing"] = df_grouped["unique_datasets"].apply(
            lambda d: mark_shared(d, genus)
        )

        # --- Merge back with TCR-level data ---
        merged_df = df.merge(
            df_grouped[["cluster", "color_sharing"]], on="cluster", how="left"
        )
        merged_df["genus"] = genus

        if not merged_df.empty:
            all_clustered_dfs.append(merged_df)
            red_only_dfs.append(merged_df[merged_df["color_sharing"] == "red"])

        # --- Ranking clusters ---
        x = df_grouped["mean_convergence"]
        y = df_grouped["unique_patients"]
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
        df_grouped["topright_score"] = x_norm + y_norm
        df_grouped["rank"] = rankdata(-df_grouped["topright_score"], method="average")

        # --- Rank-sum test for publicity ---
        red_ranks = df_grouped[df_grouped["color_sharing"] == "red"]["rank"]
        grey_ranks = df_grouped[df_grouped["color_sharing"] != "red"]["rank"]
        p_value = (
            mannwhitneyu(red_ranks, grey_ranks, alternative="less")[1]
            if len(red_ranks) > 0 and len(grey_ranks) > 0
            else np.nan
        )
        raw_pvals.append(p_value)

        overview_data.append(
            [
                genus,
                (df_grouped["color_sharing"] == "lightgrey").sum(),
                (df_grouped["color_sharing"] == "red").sum(),
                p_value,
                np.nan,  # placeholder for adjusted p
            ]
        )

        # --- Shared clusters in top X% ---
        if len(df_grouped) > 0:
            top_threshold = np.percentile(df_grouped["rank"], top_percentile)
            frac_top = (
                df_grouped.loc[df_grouped["color_sharing"] == "red", "rank"]
                <= top_threshold
            ).mean()
            summary_frac_data.append({"Genus": genus, "FractionTop": frac_top})

            all_ranks_data.extend(
                [
                    {"rank": r, "is_red": (c == "red"), "genus": genus}
                    for r, c in zip(df_grouped["rank"], df_grouped["color_sharing"])
                ]
            )

        grouped_data_list.append(df_grouped)

    # --- Multiple testing correction ---
    raw_pvals_array = np.array(raw_pvals)
    valid_mask = ~np.isnan(raw_pvals_array)
    adj_pval_array = np.full_like(raw_pvals_array, np.nan, dtype=float)
    adj_pval_array[valid_mask] = multipletests(
        raw_pvals_array[valid_mask], method="fdr_bh"
    )[1]

    for i in range(len(overview_data)):
        overview_data[i][4] = adj_pval_array[i]

    # --- Save outputs ---
    overview_df = pd.DataFrame(
        overview_data,
        columns=[
            "Genus",
            "# publicity <3",
            "# publicity = 3",
            "Raw p-value",
            "Adjusted p-value",
        ],
    )
    overview_df.to_csv(output_dir / "2_overview_df.csv", index=False)

    if red_only_dfs:
        pd.concat(all_clustered_dfs, ignore_index=True).to_csv(
            output_dir / "3_all_cluster_TCR.csv", index=False
        )
        pd.concat(red_only_dfs, ignore_index=True).to_csv(
            output_dir / "4_red_cluster_TCR.csv", index=False
        )

    return {
        "overview": overview_df,
        "all_clustered": all_clustered_dfs,
        "red_only": red_only_dfs,
        "adjusted_pval": adj_pval_array,
        "summary_frac": pd.DataFrame(summary_frac_data),
        "all_ranks": pd.DataFrame(all_ranks_data),
        "grouped": grouped_data_list,
    }


# ============================================================================================== #
### Plotting functions ###
# ============================================================================================== #

##############################################
### Get genus prevalence in discovery data ###
##############################################


def count_genus_prevalence(genus_df, genus):
    patients_with_genus = (genus_df[genus] != 0).sum()
    patients_without_genus = (genus_df[genus] == 0).sum()
    return patients_with_genus, patients_without_genus


#############################################
### Violin plot convergence and publicity ###
#############################################


def sample_violin_plot(ax, df, x, y, hue, palette, xtick_labels, title):
    """
    Draws a violin first, then grey & colored points on top.
    """
    grey = df[df[hue] == "#bebebc"]
    colored = df[df[hue] != "#bebebc"]

    sns.violinplot(
        data=df,
        x=x,
        y=y,
        color="lightgrey",
        fill=False,
        inner=None,
        cut=0,
        scale="width",
        width=0.6,
        ax=ax,
        zorder=0,
        alpha=1,
    )

    sns.stripplot(
        data=grey,
        x=x,
        y=y,
        color="lightgrey",
        linewidth=1,
        edgecolor="white",
        jitter=0.15,
        size=8,
        ax=ax,
        zorder=1,
    )

    sns.stripplot(
        data=colored,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        linewidth=1,
        edgecolor="white",
        jitter=0.15,
        size=8,
        ax=ax,
        zorder=2,
    )

    ax.set_xticklabels([xtick_labels[val] for val in df[x].unique()])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    sns.despine(ax=ax, top=True, right=True)
    ax.legend([], [], frameon=False)


######################################################
### T cell subtype distribution in convergent TCRs ###
######################################################


def subtype_overview(
    full_tcr_info,
    before_df,
    sorted_genera,
    dataset_name,
    group_col="cell_type",
    aggregate_col=None,
):
    """
    Generalized overview builder:
    - Compares convergent vs baseline counts per genus
    - Can group by any column (e.g. 'cell_type', 'run_id')
    - Optionally also adds aggregated results by another column (e.g. cohort cell_type)
    - If `keys` provided, will hash them for baseline filtering
    """
    results = []

    for genus in sorted_genera:
        conv_df = full_tcr_info[
            (full_tcr_info["genus"] == genus) & (full_tcr_info["convergent"] == "Yes")
        ]
        if conv_df.empty:
            continue

        baseline_counts = before_df[group_col].value_counts()
        conv_counts = conv_df[group_col].value_counts()
        all_groups = sorted(set(baseline_counts.index) | set(conv_counts.index))

        table = [
            [conv_counts.get(g, 0) for g in all_groups],
            [baseline_counts.get(g, 0) for g in all_groups],
        ]

        if len(all_groups) == 2:
            _, pvalue = fisher_exact(table)
        else:
            _, pvalue, _, _ = chi2_contingency(table)

        total_conv = conv_counts.sum()
        total_base = baseline_counts.sum()

        for g in all_groups:
            conv = conv_counts.get(g, 0)
            frac_conv = conv / total_conv if total_conv > 0 else 0
            frac_base = baseline_counts.get(g, 0) / total_base if total_base > 0 else 0
            fold_change = frac_conv / frac_base if frac_base > 0 else np.nan
            log2fc = (
                np.log2(fold_change)
                if (fold_change > 0 and np.isfinite(fold_change))
                else 0
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "genus": genus,
                    group_col: g,
                    "ConvergentCount": conv,
                    "BaselineCount": baseline_counts.get(g, 0),
                    "FoldChange": fold_change,
                    "Log2FC": log2fc,
                    "PValue": pvalue,
                }
            )

        # Optional aggregated grouping across all patients (for exp 2.)
        if aggregate_col is not None:
            agg_conv_counts = conv_df[aggregate_col].value_counts()
            agg_baseline_counts = before_df[aggregate_col].value_counts()
            agg_total_conv = agg_conv_counts.sum()
            agg_total_base = agg_baseline_counts.sum()

            for agg in sorted(
                set(agg_conv_counts.index) | set(agg_baseline_counts.index)
            ):
                conv = agg_conv_counts.get(agg, 0)
                frac_conv = conv / agg_total_conv if agg_total_conv > 0 else 0
                frac_base = (
                    agg_baseline_counts.get(agg, 0) / agg_total_base
                    if agg_total_base > 0
                    else 0
                )
                fold_change = frac_conv / frac_base if frac_base > 0 else np.nan
                log2fc = (
                    np.log2(fold_change)
                    if (fold_change > 0 and np.isfinite(fold_change))
                    else 0
                )

                results.append(
                    {
                        "dataset": dataset_name,
                        "genus": genus,
                        group_col: f"{agg}_agg",
                        "ConvergentCount": conv,
                        "BaselineCount": agg_baseline_counts.get(agg, 0),
                        "FoldChange": fold_change,
                        "Log2FC": log2fc,
                        "PValue": np.nan,  # aggregated, no p-value
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        df["negLog10P"] = -np.log10(df["PValue"].fillna(1) + 1e-10)
    return df
