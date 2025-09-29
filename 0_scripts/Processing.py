# Imports
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, kruskal
import scikit_posthocs as sp
import itertools


# ============================================================================================== #
### Data processing functions ###
# ============================================================================================== #

# Set of standard amino acids
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


####################
### Data loading ###
####################


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


####################################################
### Load and process the IBD data (Brand et al.) ###
####################################################


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


# ============================================================================================== #
### Plotting functions ###
# ============================================================================================== #

##########################
### Diversity boxplots ###
##########################


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


##############################################
### Statistical tests for comparing groups ###
##############################################


def _friedman_wilcoxon(
    df_values,
    value_col="Value",
    group_col="Condition",
    patient_col="Patient",
    p_adjust="fdr_bh",
):
    """Run Friedman test with Wilcoxon posthoc (paired/repeated measures)."""
    df_wide = df_values.pivot(
        index=patient_col, columns=group_col, values=value_col
    ).dropna()
    if df_wide.shape[1] < 2:
        return None, None

    stat, p = friedmanchisquare(*[df_wide[c] for c in df_wide.columns])

    df_long_tmp = df_wide.reset_index().melt(
        id_vars=patient_col,
        value_vars=df_wide.columns,
        var_name=group_col,
        value_name=value_col,
    )
    posthoc = sp.posthoc_wilcoxon(
        df_long_tmp,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust,
    )
    return (stat, p), posthoc


def _kruskal_dunn(
    df_values,
    value_col="Value",
    group_col="Condition",
    p_adjust="fdr_bh",
):
    """Run Kruskal–Wallis test with Dunn posthoc (independent groups)."""
    groups_in_data = df_values[group_col].dropna().unique()
    if len(groups_in_data) < 2:
        return None, None

    grouped = [
        df_values.loc[df_values[group_col] == g, value_col].dropna()
        for g in groups_in_data
    ]
    stat, p = kruskal(*grouped)

    posthoc = sp.posthoc_dunn(
        df_values,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust,
    )
    return (stat, p), posthoc.astype(float)


########################################
### Compare diversity between groups ###
########################################


def diversity_statistics(
    df_values,
    metric_label,
    test="kruskal",
    value_col="Value",
    group_col="Condition",
    patient_col="Patient",
    p_adjust="fdr_bh",
):
    """Run chosen statistical test and return long-format results table."""
    if test == "friedman":
        omnibus, posthoc = _friedman_wilcoxon(
            df_values, value_col, group_col, patient_col, p_adjust
        )
        test_name = "Friedman + Wilcoxon (FDR)"
    elif test == "kruskal":
        omnibus, posthoc = _kruskal_dunn(df_values, value_col, group_col, p_adjust)
        test_name = "Kruskal–Wallis + Dunn (FDR)"
    else:
        raise ValueError("test must be 'friedman' or 'kruskal'")

    if omnibus is None or posthoc is None:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Test",
                "Main_stat",
                "Main_p",
                "Group1",
                "Group2",
                "posthoc_p_adj",
            ]
        )

    stat, p = omnibus
    records = []

    groups = list(posthoc.columns)
    for g1, g2 in itertools.combinations(groups, 2):
        p_adj = posthoc.loc[g1, g2]
        records.append(
            {
                "Metric": metric_label,
                "Test": test_name,
                "Main_stat": stat,
                "Main_p": p,
                "Group1": g1,
                "Group2": g2,
                "posthoc_p_adj": float(p_adj) if pd.notnull(p_adj) else None,
            }
        )

    return pd.DataFrame(records)


########################################
### Annotate significant differences ###
########################################


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
    patient_col="Patient",
    x_base=0,
    box_width=0.8,
    test="kruskal",
    p_adjust="fdr_bh",
):
    """Annotate plots with significant pairwise comparisons."""
    if test == "friedman":
        _, posthoc = _friedman_wilcoxon(
            df_values.rename(columns={value_col: "Value"}),
            "Value",
            group_col,
            patient_col,
            p_adjust,
        )
    elif test == "kruskal":
        _, posthoc = _kruskal_dunn(
            df_values.rename(columns={value_col: "Value"}),
            "Value",
            group_col,
            p_adjust,
        )
    else:
        return

    if posthoc is None:
        return

    max_y = float(df_values[value_col].max())
    count = 0
    present_groups = list(posthoc.columns)
    n_conditions_plot = len(present_groups)

    for g1, g2 in itertools.combinations(present_groups, 2):
        p_val = posthoc.loc[g1, g2]
        if pd.notnull(p_val) and p_val < 0.05:
            # Align brackets with the groups
            x1_idx = (
                x_base
                - box_width / 2
                + conditions.index(g1) * (box_width / n_conditions_plot)
                + (box_width / n_conditions_plot) / 2
            )
            x2_idx = (
                x_base
                - box_width / 2
                + conditions.index(g2) * (box_width / n_conditions_plot)
                + (box_width / n_conditions_plot) / 2
            )
            y_val = max_y + count * 0.05 * max_y if max_y > 0 else (count + 1) * 0.05
            _add_stat_annotation(ax, x1_idx, x2_idx, y_val, p_val)
            count += 1
