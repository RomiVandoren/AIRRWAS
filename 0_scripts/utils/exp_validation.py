import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import zscore, norm, fisher_exact, chi2_contingency
from upsetplot import UpSet, from_indicators, plot


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


################################
### Extract Mixcr statistics ###
################################


def extract_metrics_from_report(report_file):
    metrics = {}
    with open(report_file, "r") as file:
        for line in file:
            if "Total sequencing reads" in line:
                metrics["Total sequencing reads"] = int(
                    line.split(":")[1].strip().replace(",", "")
                )
            elif "Successfully aligned reads" in line:
                metrics["Successfully aligned reads"] = int(
                    line.split(":")[1].split("(")[0].strip().replace(",", "")
                )
                metrics["Successfully aligned reads (%)"] = float(
                    line.split("(")[1].split("%")[0].strip().replace(",", ".")
                )
    return metrics


def extract_metrics_from_assemble(assemble_file):
    metrics = {}
    with open(assemble_file, "r") as file:
        for line in file:
            if "Final clonotype count" in line:
                metrics["Final clonotype count"] = int(
                    line.split(":")[1].strip().replace(",", "")
                )
            elif "Reads used in clonotypes, percent of total" in line:
                metrics["Reads used in clonotypes"] = int(
                    line.split(":")[1].split("(")[0].strip().replace(",", "")
                )
                metrics["Reads used in clonotypes (%)"] = float(
                    line.split("(")[1].split("%")[0].strip().replace(",", ".")
                )
    return metrics


###################################
### Get clonal expansion ranges ###
###################################


def get_expansion(dataframe, patient, patient_id, type="frequency"):
    data = dataframe.copy()

    # Compute clone frequency per patient
    data["total_clone_count"] = data.groupby(patient)["clonecount"].transform("sum")
    data["clonefreq"] = data["clonecount"] / data["total_clone_count"]
    data = data.drop(columns=["total_clone_count"])

    if type == "absolute":
        bins = [0, 10, 100, 500, 1000, np.inf]
        bin_labels = [
            "1 < N < 10",
            "10 < N <= 100",
            "100 < N <= 500",
            "500 < N <= 1000",
            "N > 1000",
        ]
        data["Bin"] = pd.cut(
            data["clonecount"], bins=bins, labels=bin_labels, right=True
        )
    elif type == "frequency":
        bins = [0, 0.0001, 0.001, 0.01, 1]
        bin_labels = ["0% - 0.01%", "0.01% - 0.1%", "0.1% - 1%", ">1%"]
        data["Bin"] = pd.cut(
            data["clonefreq"], bins=bins, labels=bin_labels, right=True
        )

    # Group, normalize, and return
    df = (
        data.groupby(patient_id)["Bin"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
        .reset_index()
        .set_index(patient_id)
    )

    return df


##############################################################
### Get TCR overlap between stimulation and discovery data ###
##############################################################


def summarize_overlap_conv(stim_data, discovery_data, conv_data, cell_name):
    chains = ["TRA", "TRB"]

    # Stimulation data counts per patient × chain
    stim_counts = (
        stim_data.groupby(["patient_id", "chain"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="chain", values="junction_aa")
        .fillna(0)
        .astype(int)
    )

    # General overlap with the discovery data
    general_overlap = pd.merge(stim_data, discovery_data, on="junction_aa", how="inner")
    general_overlap = general_overlap.drop_duplicates(
        subset=["junction_aa", "patient_id", "patient_y"]
    )
    general_counts = (
        general_overlap.groupby(["patient_id", "chain_x"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="chain_x", values="junction_aa")
        .fillna(0)
        .astype(int)
    )
    general_counts = general_counts.rename(
        columns={"TRA": "TRA_overlap", "TRB": "TRB_overlap"}
    )

    # Patient-matched overlap with discovery data
    patient_matched = pd.merge(
        stim_data,
        discovery_data,
        left_on=["junction_aa", "patient_id"],
        right_on=["junction_aa", "patient"],
        how="inner",
    )
    patient_matched = patient_matched.drop_duplicates(
        subset=["junction_aa", "patient_id"]
    )
    patient_counts = (
        patient_matched.groupby(["patient_id", "chain_x"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="chain_x", values="junction_aa")
        .fillna(0)
        .astype(int)
    )
    patient_counts = patient_counts.rename(
        columns={"TRA": "TRA_patient_matched", "TRB": "TRB_patient_matched"}
    )

    # General overlap with CONVERGENT TCRs in the discovery data
    general_overlap_conv = pd.merge(stim_data, conv_data, on="junction_aa", how="inner")
    general_overlap_conv = general_overlap_conv.drop_duplicates(
        subset=["junction_aa", "patient_id", "patient_y"]
    )
    general_counts_conv = (
        general_overlap_conv.groupby(["patient_id", "chain_x"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="chain_x", values="junction_aa")
        .fillna(0)
        .astype(int)
    )
    general_counts_conv = general_counts_conv.rename(
        columns={"TRA": "TRA_overlap_conv", "TRB": "TRB_overlap_conv"}
    )

    # Patient-matched overlap with CONVERGENT TCRs in the discovery data
    patient_matched_conv = pd.merge(
        stim_data,
        conv_data,
        left_on=["junction_aa", "patient_id"],
        right_on=["junction_aa", "patient"],
        how="inner",
    )
    patient_matched_conv = patient_matched_conv.drop_duplicates(
        subset=["junction_aa", "patient_id"]
    )
    patient_counts_conv = (
        patient_matched_conv.groupby(["patient_id", "chain_x"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="chain_x", values="junction_aa")
        .fillna(0)
        .astype(int)
    )
    patient_counts_conv = patient_counts_conv.rename(
        columns={"TRA": "TRA_patient_matched_conv", "TRB": "TRB_patient_matched_conv"}
    )

    # Combine all tables
    summary = (
        stim_counts.join(general_counts, how="outer")
        .join(patient_counts, how="outer")
        .join(general_counts_conv, how="outer")
        .join(patient_counts_conv, how="outer")
        .fillna(0)
        .astype(int)
    )
    summary["cell_type"] = cell_name
    summary = summary.reset_index()

    # Percentages for patient-matched
    for chain in chains:
        summary[f"{chain}_patient_matched"] = summary.apply(
            lambda x: (
                f"({(x[f'{chain}_patient_matched'] / x[chain] * 100):.1f}%)"
                if x[chain] > 0
                else "0 / 0 (0.0%)"
            ),
            axis=1,
        )
        summary[f"{chain}_overlap_conv"] = summary.apply(
            lambda x: (
                f"({(x[f'{chain}_overlap_conv'] / x[f'{chain}_overlap'] * 100):.1f}%)"
                if x[chain] > 0
                else "0 / 0 (0.0%)"
            ),
            axis=1,
        )
        summary[f"{chain}_patient_matched_conv"] = summary.apply(
            lambda x: (
                f"({(x[f'{chain}_patient_matched_conv'] / x[f'{chain}_overlap'] * 100):.1f}%)"
                if x[chain] > 0
                else "0 / 0 (0.0%)"
            ),
            axis=1,
        )

    # Reorder columns
    cols_order = [
        "patient_id",
        "cell_type",
        "TRA",
        "TRB",
        "TRA_overlap",
        "TRB_overlap",
        "TRA_patient_matched",
        "TRB_patient_matched",
        "TRA_overlap_conv",
        "TRB_overlap_conv",
        "TRA_patient_matched_conv",
        "TRB_patient_matched_conv",
    ]
    summary = summary[cols_order]
    return summary


##################################
### Get TCR-level overlap data ###
##################################


def get_overlap_dataframe(df, disc_data, conv_data, prefix):
    """
    Compute overlap counts (total, ERC, convergent), merge them,
    and return a sorted dataframe + labels for plotting.
    """
    # Total unique TCR counts
    total = (
        df.groupby(["patient_id", "condition"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="condition", values="junction_aa")
        .reset_index()
    )

    # Discovery data overlap
    general_erc = pd.merge(
        df, disc_data, on="junction_aa", how="inner"
    ).drop_duplicates(subset=["junction_aa", "patient_id", "patient_y"])
    erc = (
        general_erc.groupby(["patient_id", "condition"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="condition", values="junction_aa")
        .reset_index()
    )

    # Convergent overlap
    general_conv = pd.merge(
        df, conv_data, on="junction_aa", how="inner"
    ).drop_duplicates(subset=["junction_aa", "patient_id", "patient_y"])
    conv = (
        general_conv.groupby(["patient_id", "condition"])["junction_aa"]
        .nunique()
        .reset_index()
        .pivot(index="patient_id", columns="condition", values="junction_aa")
        .reset_index()
    )

    # Melt + merge into tidy format
    def melt_and_rename(frame, value_name):
        return frame.melt(
            id_vars="patient_id", var_name="condition", value_name=value_name
        )

    merged = (
        melt_and_rename(total, "Total TCRs")
        .merge(
            melt_and_rename(erc, "Overlap with discovery"),
            on=["patient_id", "condition"],
            how="outer",
        )
        .merge(
            melt_and_rename(conv, "Overlap with convergent TCRs"),
            on=["patient_id", "condition"],
            how="outer",
        )
        .fillna(0)
    )

    # Filter and add labels
    merged = merged[merged["condition"].str.startswith(prefix)]
    merged["patient_condition"] = (
        merged["patient_id"].astype(str)
        + "_"
        + merged["condition"].str.split("_").str[1]
    )

    # Sort labels (1d before 1w)
    def sort_labels(labels):
        one_d = sorted(
            [l for l in labels if l.endswith("_1d")], key=lambda s: s.split("_")[0]
        )
        one_w = sorted(
            [l for l in labels if l.endswith("_1w")], key=lambda s: s.split("_")[0]
        )
        return one_d + one_w

    labels = sort_labels(merged["patient_condition"])
    df_sorted = (
        merged.set_index("patient_condition").reindex(labels).fillna(0).reset_index()
    )

    return df_sorted, labels


############################################
### Get overlap with convergent clusters ###
############################################


def compute_overlap_clusters_conv(
    conv_df,
    lab_df,
    condition_col,
    genus_to_check,
    remove_small=False,
    min_tcrs=1000,
    use_median=False,
):
    conv_df = conv_df.copy()
    conv_df["chain"] = conv_df["v_call"].str.extract(r"(TR[AB])")
    conv_nopat = conv_df.drop_duplicates(subset=["full_tcr", "chain", "genus"])
    lab_df = lab_df.drop_duplicates(
        subset=["full_tcr", "chain", "patient_id", condition_col]
    )

    results = []
    for chain_type in ["TRA", "TRB"]:
        lab_tcrs_set = set(lab_df[lab_df["chain"] == chain_type]["full_tcr"])
        subset = conv_nopat[conv_nopat["chain"] == chain_type]

        for genus in subset["genus"].unique():
            genus_subset = subset[subset["genus"] == genus]
            genus_tcrs_set = set(genus_subset["full_tcr"])

            overlap_tcrs = genus_tcrs_set & lab_tcrs_set
            n_overlap = len(overlap_tcrs)
            n_total = len(genus_tcrs_set)

            percent_overlap = (n_overlap / n_total) * 100 if n_total > 0 else 0

            results.append(
                {
                    "chain": chain_type,
                    "genus": genus,
                    "n_predicted_TCRs": n_total,
                    "n_overlap_with_lab": n_overlap,
                    "percent_overlap": percent_overlap,
                }
            )

    overlap_df = pd.DataFrame(results)
    overlap_df = overlap_df.sort_values(by="n_overlap_with_lab", ascending=False)

    # Optionally filter out small genera
    if remove_small:
        smallest_genera = overlap_df[overlap_df["n_predicted_TCRs"] <= min_tcrs][
            "genus"
        ].unique()
        smallest_genera = [g for g in smallest_genera if g != "Prevotella"]
        overlap_df = overlap_df[~overlap_df["genus"].isin(smallest_genera)]

    # Statistical testing per genus and chain
    stat_results = []
    for metric in ["n_overlap_with_lab", "percent_overlap"]:
        for chain in ["TRA", "TRB"]:
            subset = overlap_df[overlap_df["chain"] == chain]
            pep_row = subset[subset["genus"] == genus_to_check]

            if pep_row.empty:
                stat_results.append(
                    {
                        "metric": metric,
                        "chain": chain,
                        "peptoniphilus_value": np.nan,
                        "z_score": np.nan,
                        "p_value": np.nan,
                        "note": "Peptoniphilus not found",
                    }
                )
                continue

            val = pep_row[metric].values[0]
            other_vals = subset[subset["genus"] != genus_to_check][metric].values

            # Calculate Z-score
            if use_median:
                median = np.median(other_vals)
                iqr = np.percentile(other_vals, 75) - np.percentile(other_vals, 25)
                z = (val - median) / iqr if iqr > 0 else np.nan
            else:
                mean = np.mean(other_vals)
                std = np.std(other_vals)
                z = (val - mean) / std if std > 0 else np.nan

            # Calculate p-value for the one-tailed test
            p_val = 1 - norm.cdf(z) if not np.isnan(z) else np.nan

            stat_results.append(
                {
                    "metric": metric,
                    "chain": chain,
                    "peptoniphilus_value": val,
                    "z_score": z,
                    "p_value": p_val,
                }
            )

    stat_df = pd.DataFrame(stat_results)

    return overlap_df, stat_df


# ============================================================================================== #
### Plotting functions ###
# ============================================================================================== #


################################
### Get Mixcr stats overview ###
################################


def plot_violin_outline(ax, conditions, violin_data, label_map):
    sns.violinplot(
        data=violin_data[violin_data["condition"].isin(conditions)],
        x="condition",
        y="Successfully aligned reads (%)",
        hue="condition",
        ax=ax,
        inner=None,
        fill=False,  # outline only
        palette=["#669bbc", "#FF858D"],
    )

    sns.stripplot(
        x="condition",
        y="Successfully aligned reads (%)",
        data=violin_data[violin_data["condition"].isin(conditions)],
        color="k",
        size=4,
        jitter=True,
        ax=ax,
    )

    ax.set_ylim(-30, 85)
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([label_map[c] for c in conditions])
    ax.set_ylabel("% aligned", fontsize=14)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bars_twin(ax, conditions, grouped_data, label_map, chains, palette_tcr):
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_bar = ax.twinx()
    subset = grouped_data[grouped_data["condition"].isin(conditions)]
    patients = sorted(subset["patient"].unique())
    x = np.arange(len(patients))
    bar_width = 0.35

    for i, condition in enumerate(conditions):
        df = subset[subset["condition"] == condition]
        x_shift = [-bar_width / 2, bar_width / 2][i]
        bottom = np.zeros(len(patients))
        for chain in chains:
            heights = [
                (
                    df[df["patient"] == p][chain].values[0]
                    if p in df["patient"].values
                    else 0
                )
                for p in patients
            ]
            ax_bar.bar(
                x + x_shift,
                heights,
                bar_width,
                bottom=bottom,
                color=palette_tcr[f"{condition}_{chain}"],
                label=f"{chain} - {label_map[condition]}",
            )
            bottom += heights

    ax_bar.yaxis.set_label_position("right")
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.set_ylim(0, 3200)
    ax_bar.set_ylabel("Unique TCR count", fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(patients, rotation=45, ha="right")
    ax_bar.legend(frameon=False, fontsize=11)


#########################################
### Get Upset plot of the public TCRs ###
#########################################


def Upset_public_tcrs(
    df,
    suppl_dir,
    condition_prefix,
    sample_col="patient_id",
    chain_col="chain",
    junction_col="junction_aa",
    min_shared=3,
    figsize=(12, 6),
):

    subset_df = df[df["condition"].str.startswith(condition_prefix)].copy()
    subset_df["chain_tcr"] = subset_df[chain_col] + "_" + subset_df[junction_col]
    binary_matrix = pd.crosstab(subset_df["chain_tcr"], subset_df[sample_col]).astype(
        bool
    )

    # Upset plot
    upset_data = from_indicators(binary_matrix.columns, binary_matrix)
    fig = plt.figure(figsize=figsize)
    plot(upset_data, fig=fig, element_size=40, show_counts=True, min_degree=2)

    plt.tight_layout()
    plt.savefig(
        suppl_dir / f"S4_validation1_upset_{condition_prefix}.pdf", bbox_inches="tight"
    )
    plt.show()

    # Extract public TCRs (shared across ≥ min_shared patients)
    shared_tcrs = binary_matrix[binary_matrix.sum(axis=1) >= min_shared]
    sample_names = shared_tcrs.columns.tolist()
    publicity = shared_tcrs.sum(axis=1)
    patients = shared_tcrs.apply(
        lambda row: ",".join(
            [sample_names[i] for i, present in enumerate(row) if present]
        ),
        axis=1,
    )

    public_df = pd.DataFrame(
        {
            "chain_tcr_id": shared_tcrs.index,
            "publicity": publicity.values,
            "patients": patients.values,
        }
    ).reset_index(drop=True)

    public_df[["chain", "junction_aa"]] = public_df["chain_tcr_id"].str.split(
        "_", expand=True
    )
    public_df = (
        public_df[["chain", "junction_aa", "publicity", "patients"]]
        .sort_values(by="publicity", ascending=False)
        .reset_index(drop=True)
    )

    return public_df


#########################################################################
### Create barplot for overlap between stimulation and discovery data ###
#########################################################################


def add_timepoint_labels(ax, labels, y_offset=-0.25):
    groups = [lab.split("_")[1] for lab in labels]
    x = np.arange(len(labels))
    label_map = {"1d": "1 day stimulation", "1w": "1 week stimulation"}

    for tp in sorted(set(groups)):
        idxs = [i for i, g in enumerate(groups) if g == tp]
        if not idxs:
            continue
        start, end = min(idxs), max(idxs)
        mid = (start + end) / 2

        ax.text(
            mid,
            y_offset,
            label_map.get(tp, tp),
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=14,
        )

        ax.plot(
            [start - 0.2, end],
            [y_offset + 0.02, y_offset + 0.02],
            color="black",
            linewidth=1,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

        for pos in [start - 0.2, end]:
            ax.plot(
                [pos, pos],
                [y_offset + 0.015, y_offset + 0.025],
                color="black",
                linewidth=1,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )


def format_axes(ax, title, ylabel=None, xticklabels=None, legend=False):
    ax.set_title(title, size=18, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, size=15)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, size=13)
    ax.tick_params(axis="y", labelsize=12)
    if legend:
        ax.legend(frameon=False, fontsize=15, bbox_to_anchor=(1.1, 0.95))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.margins(x=0)


def plot_stim_overlap(ax, df_sorted, labels, title, legend=False):
    groups = [
        ("Total TCRs", "#BABABA"),
        ("Overlap with discovery", "#4C7778"),
        ("Overlap with convergent TCRs", "#E17A89"),
    ]
    x = np.arange(len(labels))

    for name, color in groups:
        ax.bar(x, df_sorted[name], width=0.8, label=name, color=color)

    format_axes(
        ax,
        title,
        ylabel="Number of TCRs",
        xticklabels=[lab.split("_")[0] for lab in labels],
        legend=legend,
    )
    add_timepoint_labels(ax, labels)


#####################################################################
### Plot relative overlap with genus-specific convergent clusters ###
#####################################################################


def plot_relative_overlap(
    panels: dict,
    metric: str = "percent_overlap",
    savename: str | None = None,
):
    """
    Draws violin/strip plots comparing overlap metrics across panels.

    Parameters
    ----------
    panels : dict
        Dictionary where keys are panel titles (str), and values are tuples:
        (overlap_df, stats_df, genus_to_highlight).
    metric : str
        Column name in overlap_df to plot (default: 'percent_overlap').
    savename : str or None
        If given, save figure to this path (PDF/PNG).
    """
    sns.set(style="white")
    chain_order = ["TRA", "TRB"]
    colors = {"TRA": "#e6a272", "TRB": "#9db7d6"}

    # enforce categorical ordering for chain
    for overlap_df, _, _ in panels.values():
        overlap_df["chain"] = pd.Categorical(
            overlap_df["chain"], categories=chain_order, ordered=True
        )

    # make subplots
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 5), gridspec_kw={"wspace": 0.3}
    )

    if len(panels) == 1:
        axes = [axes]  # keep iterable

    for ax, (title, (overlap_df, stats_df, genus_to_highlight)) in zip(
        axes, panels.items()
    ):
        # violin + strip
        sns.violinplot(
            data=overlap_df,
            x="chain",
            y=metric,
            palette=colors,
            inner=None,
            width=0.8,
            order=chain_order,
            ax=ax,
        )
        sns.stripplot(
            data=overlap_df,
            x="chain",
            y=metric,
            color="black",
            size=4,
            jitter=True,
            alpha=0.6,
            order=chain_order,
            ax=ax,
        )

        for chain_type in chain_order:
            x_pos = chain_order.index(chain_type)

            # highlight genus point
            val = overlap_df.loc[
                (overlap_df["chain"] == chain_type)
                & (overlap_df["genus"] == genus_to_highlight),
                metric,
            ]
            if not val.empty:
                y_val = val.values[0]
                ax.scatter(
                    x_pos, y_val, color="red", edgecolor="black", s=100, zorder=3
                )
                ax.annotate(
                    f"{genus_to_highlight}: {y_val:.2f}%",
                    xy=(x_pos, y_val),
                    xytext=(10, 20),
                    textcoords="offset points",
                    fontsize=12,
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
                )

            # add p-value if available
            p_row = stats_df.query("metric == @metric and chain == @chain_type")
            if not p_row.empty and pd.notna(p_row["p_value"].values[0]):
                p_val = p_row["p_value"].values[0]
                ymax = overlap_df[metric].max()
                ypad = 0.45 * ymax if ymax > 0 else 0.5
                ax.text(
                    x_pos,
                    ymax + ypad,
                    f"p = {p_val:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontstyle="italic",
                )

        # styling
        ax.set_title(title, fontsize=14, weight="bold", pad=40)
        ax.set_xlabel("")
        ax.set_ylabel("% of Convergent TCRs", fontsize=12)
        ax.set_xticks(range(len(chain_order)))
        ax.set_xticklabels(chain_order)
        sns.despine(ax=ax, top=True, right=True)

    # legend
    handles = [Patch(color=c, label=ch) for ch, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=12)

    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches="tight")
    plt.show()


######################################################
### T cell subtype distribution in convergent TCRs ###
######################################################


def build_overview(full_tcr_info, before_df, sorted_genera, dataset_name):
    results = []

    for genus in sorted_genera:
        conv_df = full_tcr_info[
            (full_tcr_info["genus"] == genus) & (full_tcr_info["convergent"] == "Yes")
        ]
        if conv_df.empty:
            continue

        baseline_counts = before_df["run_id"].value_counts()
        conv_counts = conv_df["run_id"].value_counts()

        all_runs = sorted(set(baseline_counts.index) | set(conv_counts.index))
        table = [
            [conv_counts.get(r, 0) for r in all_runs],
            [baseline_counts.get(r, 0) for r in all_runs],
        ]

        if len(all_runs) == 2:
            _, pvalue = fisher_exact(table)
        else:
            _, pvalue, _, _ = chi2_contingency(table)

        total_conv = conv_counts.sum()
        total_base = baseline_counts.sum()

        # per-run rows
        for run in all_runs:
            conv = conv_counts.get(run, 0)
            frac_conv = conv / total_conv if total_conv > 0 else 0
            frac_base = (
                baseline_counts.get(run, 0) / total_base if total_base > 0 else 0
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
                    "cell_type": run,
                    "ConvergentCount": conv,
                    "BaselineCount": baseline_counts.get(run, 0),
                    "FoldChange": fold_change,
                    "Log2FC": log2fc,
                    "PValue": pvalue,
                }
            )

        # aggregated cohort-level rows
        agg_conv_counts = conv_df["cell_type"].value_counts()
        agg_baseline_counts = before_df["cell_type"].value_counts()
        agg_total_conv = agg_conv_counts.sum()
        agg_total_base = agg_baseline_counts.sum()

        for cell_type in sorted(
            set(agg_conv_counts.index) | set(agg_baseline_counts.index)
        ):
            conv = agg_conv_counts.get(cell_type, 0)
            frac_conv = conv / agg_total_conv if agg_total_conv > 0 else 0
            frac_base = (
                agg_baseline_counts.get(cell_type, 0) / agg_total_base
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
                    "cell_type": f"{cell_type}_agg",  # aggregated row marker
                    "ConvergentCount": conv,
                    "BaselineCount": agg_baseline_counts.get(cell_type, 0),
                    "FoldChange": fold_change,
                    "Log2FC": log2fc,
                    "PValue": np.nan,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df["negLog10P"] = -np.log10(df["PValue"] + 1e-10)
    return df
