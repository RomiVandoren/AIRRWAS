# Standard library
import re
import itertools
from pathlib import Path

# Third-party scientific stack
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal
import scikit_posthocs as sp

# ============================= #
### Data processing functions ###
# ============================= #

# Set of standard amino acids
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


### Data loading ###


def make_patient_dataframe(
    directory: str | Path, patient: str, cell_type: str
) -> pd.DataFrame:
    """
    Load and concatenate all TCR files for a given patient and cell type.

    Parameters
    ----------
    directory : str or Path
        Path to the directory containing raw TCR files.
    patient : str
        Patient ID string to match in filenames.
    cell_type : str
        Cell type string (e.g., "CD4" or "CD8").

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all matching files.
        Returns empty DataFrame if no files are found.
    """
    directory = Path(directory)
    pattern = re.compile(rf".*[_-]{patient}[_-]{cell_type}.*")

    dfs = []
    for file in directory.iterdir():
        if file.is_file() and pattern.match(file.name):
            df = pd.read_csv(file, sep="\t")
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["patient"] = patient
    df_all["cell_type"] = cell_type
    return df_all


### TCR parsing functions ###


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


def parse_mixcr(
    df: pd.DataFrame, imgt_file: str | Path, rename: bool = True
) -> pd.DataFrame:
    """
    Parse MiXCR output and filter for functional CDR3 sequences.

    Parameters
    ----------
    df : pd.DataFrame
        MiXCR output table.
    imgt_file : str or Path
        Path to IMGT reference annotation file.
    rename : bool
        Whether to rename MiXCR columns to standard schema.

    Returns
    -------
    pd.DataFrame
        Filtered and annotated TCR dataframe.
    """
    imgt = pd.read_csv(Path(imgt_file), sep="\t")
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

    df = _select_gene(df, "v_call")
    df = df[df["v_call"].isin(functional["imgt_allele_name"])]

    df = _select_gene(df, "j_call")
    df = df[df["j_call"].isin(functional["imgt_allele_name"])]

    df = df[df["junction_aa"].apply(_is_cdr3)]

    return df


### Load and process the IBD data (Brand et al.) ###


def get_IBD_data(
    data_dir: str | Path, meta_file: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IBD TCR repertoire data and merge with metadata.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing MiXCR clone files (ending with '_clones_all.txt').
    meta_file : str or Path
        Path to metadata excel file.

    Returns
    -------
    merged : pd.DataFrame
        Merged TCR + metadata DataFrame.
    """
    data_dir = Path(data_dir)
    dfs = []
    samples, unique_aa, total_fraction = [], [], []

    for file in data_dir.glob("*_clones_all.txt"):
        df = pd.read_csv(file, sep="\t")
        sample_name = file.stem.split("_")[0]
        df["Sample"] = sample_name
        dfs.append(df)

        samples.append(sample_name)
        unique_aa.append(df["aaSeqCDR3"].nunique())
        total_fraction.append(df["cloneFraction"].sum())

    result_df = pd.concat(dfs, ignore_index=True)

    metadata = pd.read_excel(meta_file)
    metadata["sample_name"] = metadata["Sample"].str.split("_").str[0]

    meta_sample_map = dict(zip(metadata["sample_name"], metadata["Record.Id"]))
    meta_celltype_map = dict(zip(metadata["sample_name"], metadata["Celltype"]))

    result_df["patient_id"] = result_df["Sample"].map(meta_sample_map)
    result_df["cell_type"] = result_df["Sample"].map(meta_celltype_map)

    meta_selection = metadata[
        [
            "sample_name",
            "Record.Id",
            "Sex",
            "Age",
            "zyg_def",
            "Concordancy",
            "Diagnosis",
            "Inflammation_rectoscopy",
            "Total_PBMCs_isolated",
        ]
    ]

    merged = result_df.merge(meta_selection, left_on="Sample", right_on="sample_name")
    return merged


# ====================== #
### Plotting functions ###
# ====================== #


### Diversity boxplots ###


def plot_div_boxplots(
    ax,
    df,
    x,
    y,
    hue,
    palette,
    dodge=True,
    jitter=True,
    title="",
    xlabel="",
    ylabel="",
    xticklabels=None,
    show_legend=True,
):
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        fliersize=0,
        width=0.8,
        dodge=dodge,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        dodge=dodge,
        color="black",
        size=2,
        alpha=0.8,
        jitter=jitter,
        ax=ax,
        legend=False,
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xticklabels:
        ax.set_xticklabels(xticklabels, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    sns.despine(ax=ax)
    # Remove legends from individual plots
    if not show_legend:
        ax.get_legend().remove()


### Statistical tests for comparing groups ###


def _kruskal_dunn(
    df_values,
    value_col="Value",
    group_col="Condition",
    p_adjust="fdr_bh",
):
    """Run Kruskal–Wallis test with Dunn posthoc (independent groups)."""
    groups = df_values[group_col].dropna().unique()
    if len(groups) < 2:
        return None, None

    grouped = [
        df_values.loc[df_values[group_col] == g, value_col].dropna() for g in groups
    ]

    stat, p = kruskal(*grouped)

    posthoc = sp.posthoc_dunn(
        df_values,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust,
    ).astype(float)

    return (stat, p), posthoc.astype(float)


### Compare diversity between groups ###


def run_stats_table_for_metric(
    df_values,
    metric_label,
    value_col="Value",
    group_col="Condition",
    p_adjust="fdr_bh",
):
    omnibus, posthoc = _kruskal_dunn(
        df_values,
        value_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust,
    )

    if omnibus is None or posthoc is None:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Test",
                "Omnibus_stat",
                "Omnibus_p",
                "Group1",
                "Group2",
                "p_value_adj",
            ]
        )

    stat, p = omnibus
    records = []

    for g1, g2 in itertools.combinations(posthoc.columns, 2):
        records.append(
            {
                "Metric": metric_label,
                "Test": "Kruskal–Wallis + Dunn (FDR)",
                "Omnibus_stat": stat,
                "Omnibus_p": p,
                "Group1": g1,
                "Group2": g2,
                "p_value_adj": posthoc.loc[g1, g2],
            }
        )

    return pd.DataFrame(records)


### Annotate significant differences ###


def _add_stat_annotation(ax, x1, x2, y, p_val):
    """Draw statistical annotation brackets with stars."""
    h = 0.03 * max(y, 1e-9)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")

    if p_val is None or pd.isna(p_val):
        stars = "ns"
    elif p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    elif p_val < 0.05:
        stars = "*"
    else:
        stars = "ns"

    ax.text(
        (x1 + x2) / 2,
        y + h * 1.3,
        stars,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def annotate_values(
    ax,
    df_values,
    conditions,
    value_col="Value",
    group_col="Condition",
    x_base=0,
    box_width=0.8,
    p_adjust="fdr_bh",
):
    omnibus, posthoc = _kruskal_dunn(
        df_values.rename(columns={value_col: "Value"}),
        value_col="Value",
        group_col=group_col,
        p_adjust=p_adjust,
    )

    if posthoc is None:
        return

    max_y = float(df_values[value_col].max())
    present_groups = list(posthoc.columns)
    n_groups = len(present_groups)
    count = 0

    for g1, g2 in itertools.combinations(present_groups, 2):
        p_val = posthoc.loc[g1, g2]
        if pd.notnull(p_val) and p_val < 0.05:
            x1 = (
                x_base
                - box_width / 2
                + conditions.index(g1) * (box_width / n_groups)
                + (box_width / n_groups) / 2
            )
            x2 = (
                x_base
                - box_width / 2
                + conditions.index(g2) * (box_width / n_groups)
                + (box_width / n_groups) / 2
            )

            y = max_y * (1 + 0.05 * count)
            _add_stat_annotation(ax, x1, x2, y, p_val)
            count += 1
