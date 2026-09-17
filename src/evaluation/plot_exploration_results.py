from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 42


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "exploration_cf"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

WORLD_PATH = (
    RESULTS_DIR
    / "world_metrics.csv"
)

PAIRWISE_PATH = (
    RESULTS_DIR
    / "pairwise_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "summary.csv"
)


# ---------------------------------------------------------
# BOOTSTRAP REPRODUCIBILITY METRICS
# ---------------------------------------------------------


def bootstrap_pairwise_metric(
    condition,
    metric,
    num_worlds,
    seed,
):
    """
    Estimate uncertainty in pairwise reproducibility metrics
    by resampling whole worlds.

    Pairwise measurements are not independent because each
    world occurs in many pairs. Therefore we bootstrap world
    IDs rather than individual pairwise observations.

    If the same original world is selected twice in a
    bootstrap sample, that self-pair is excluded rather than
    assigning it perfect similarity.
    """

    matrix = np.full(
        (
            num_worlds,
            num_worlds,
        ),
        np.nan,
        dtype=float,
    )

    for _, row in condition.iterrows():

        i = int(
            row["world_a"]
        )

        j = int(
            row["world_b"]
        )

        value = float(
            row[metric]
        )

        matrix[
            i,
            j
        ] = value

        matrix[
            j,
            i
        ] = value

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = []

    for _ in range(
        BOOTSTRAP_SAMPLES
    ):

        sampled_worlds = rng.integers(
            0,
            num_worlds,
            size=num_worlds,
        )

        sampled_matrix = matrix[
            np.ix_(
                sampled_worlds,
                sampled_worlds,
            )
        ]

        upper_triangle = np.triu_indices(
            num_worlds,
            k=1,
        )

        values = sampled_matrix[
            upper_triangle
        ]

        # Remove self-pairs created when the same original
        # world is sampled more than once.
        values = values[
            ~np.isnan(
                values
            )
        ]

        if len(
            values
        ) > 0:

            bootstrap_means.append(
                values.mean()
            )

    bootstrap_means = np.asarray(
        bootstrap_means,
        dtype=float,
    )

    # Original observed statistic.
    observed_mean = condition[
        metric
    ].mean()

    # Bootstrap standard error based on world-level
    # resampling.
    bootstrap_se = bootstrap_means.std(
        ddof=1
    )

    ci_low = (
        observed_mean
        - 1.96
        * bootstrap_se
    )

    ci_high = (
        observed_mean
        + 1.96
        * bootstrap_se
    )

    # Keep intervals inside the metric's possible range.
    if metric == "rank_correlation":

        ci_low = max(
            -1.0,
            ci_low,
        )

        ci_high = min(
            1.0,
            ci_high,
        )

    elif metric == "top_k_overlap":

        ci_low = max(
            0.0,
            ci_low,
        )

        ci_high = min(
            1.0,
            ci_high,
        )

    return (
        ci_low,
        ci_high,
    )
# ---------------------------------------------------------
# ADD BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------

def add_reproducibility_intervals(
    summary_df,
    pairwise_df,
):

    rank_low = []
    rank_high = []

    top_low = []
    top_high = []

    for index, row in (
        summary_df.iterrows()
    ):

        epsilon = row[
            "epsilon"
        ]

        num_worlds = int(
            row[
                "num_worlds"
            ]
        )

        condition = pairwise_df[
            np.isclose(
                pairwise_df[
                    "epsilon"
                ],
                epsilon,
            )
        ]

        (
            rank_ci_low,
            rank_ci_high,
        ) = bootstrap_pairwise_metric(
            condition,
            "rank_correlation",
            num_worlds,
            BOOTSTRAP_SEED
            + index,
        )

        (
            top_ci_low,
            top_ci_high,
        ) = bootstrap_pairwise_metric(
            condition,
            "top_k_overlap",
            num_worlds,
            BOOTSTRAP_SEED
            + 100
            + index,
        )

        rank_low.append(
            rank_ci_low
        )

        rank_high.append(
            rank_ci_high
        )

        top_low.append(
            top_ci_low
        )

        top_high.append(
            top_ci_high
        )

    summary_df = (
        summary_df.copy()
    )

    summary_df[
        "rank_correlation_ci_low"
    ] = rank_low

    summary_df[
        "rank_correlation_ci_high"
    ] = rank_high

    summary_df[
        "top_k_overlap_ci_low"
    ] = top_low

    summary_df[
        "top_k_overlap_ci_high"
    ] = top_high

    return summary_df


# ---------------------------------------------------------
# GENERAL LINE PLOT
# ---------------------------------------------------------

def plot_metric(
    summary_df,
    mean_column,
    low_column,
    high_column,
    ylabel,
    title,
    filename,
):

    x = summary_df[
        "epsilon"
    ].to_numpy()

    y = summary_df[
        mean_column
    ].to_numpy()

    lower = np.maximum(
        0.0,
        y
        - summary_df[
            low_column
        ].to_numpy()
    )

    upper = np.maximum(
        0.0,
        summary_df[
            high_column
        ].to_numpy()
        - y
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        x,
        y,
        yerr=[
            lower,
            upper,
        ],
        marker="o",
        capsize=4,
    )

    plt.xlabel(
        "Exploration rate (epsilon)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.xticks(
        x
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# TRADE-OFF PLOT
# ---------------------------------------------------------

def plot_tradeoff(
    summary_df,
):

    x = summary_df[
        "exposure_gini_mean"
    ].to_numpy()

    y = summary_df[
        "ctr_mean"
    ].to_numpy()

    epsilons = summary_df[
        "epsilon"
    ].to_numpy()

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    for exposure_gini, ctr, epsilon in zip(
        x,
        y,
        epsilons,
    ):

        plt.annotate(
            f"ε={epsilon:.2f}",
            (
                exposure_gini,
                ctr,
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.xlabel(
        "Exposure Gini coefficient"
    )

    plt.ylabel(
        "Click-through rate"
    )

    plt.title(
        "Recommendation Performance vs Exposure Inequality"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "ctr_vs_exposure_inequality.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    world_df = pd.read_csv(
        WORLD_PATH
    )

    pairwise_df = pd.read_csv(
        PAIRWISE_PATH
    )

    summary_df = pd.read_csv(
        SUMMARY_PATH
    )

    print(
        "=============================================="
    )

    print(
        "EXPLORATION EXPERIMENT FIGURES"
    )

    print(
        "=============================================="
    )

    print(
        f"Per-world observations: "
        f"{len(world_df):,}"
    )

    print(
        f"Pairwise observations: "
        f"{len(pairwise_df):,}"
    )

    print(
        "Calculating world-level bootstrap intervals..."
    )

    summary_df = (
        add_reproducibility_intervals(
            summary_df,
            pairwise_df,
        )
    )

    enriched_summary_path = (
        RESULTS_DIR
        / "summary_with_bootstrap.csv"
    )

    summary_df.to_csv(
        enriched_summary_path,
        index=False,
    )

    print(
        f"Saved: "
        f"{enriched_summary_path}"
    )

    print()

    # -----------------------------------------------------
    # REPRODUCIBILITY
    # -----------------------------------------------------

    plot_metric(
        summary_df,
        "rank_correlation_mean",
        "rank_correlation_ci_low",
        "rank_correlation_ci_high",
        "Mean Spearman rank correlation",
        "Outcome Reproducibility Across Parallel Worlds",
        "rank_correlation_vs_epsilon.png",
    )

    plot_metric(
        summary_df,
        "top_k_overlap_mean",
        "top_k_overlap_ci_low",
        "top_k_overlap_ci_high",
        "Mean top-20 overlap",
        "Top-20 Outcome Reproducibility",
        "top20_overlap_vs_epsilon.png",
    )

    # -----------------------------------------------------
    # INEQUALITY
    # -----------------------------------------------------

    plot_metric(
        summary_df,
        "exposure_gini_mean",
        "exposure_gini_ci_low",
        "exposure_gini_ci_high",
        "Exposure Gini coefficient",
        "Exposure Inequality vs Exploration",
        "exposure_gini_vs_epsilon.png",
    )

    plot_metric(
        summary_df,
        "popularity_gini_mean",
        "popularity_gini_ci_low",
        "popularity_gini_ci_high",
        "Popularity Gini coefficient",
        "Popularity Inequality vs Exploration",
        "popularity_gini_vs_epsilon.png",
    )

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    plot_metric(
        summary_df,
        "ctr_mean",
        "ctr_ci_low",
        "ctr_ci_high",
        "Click-through rate",
        "Recommendation Performance vs Exploration",
        "ctr_vs_epsilon.png",
    )

    plot_metric(
        summary_df,
        "shown_relevance_mean",
        "shown_relevance_ci_low",
        "shown_relevance_ci_high",
        "Mean underlying response probability",
        "Underlying Relevance of Recommended Items",
        "shown_relevance_vs_epsilon.png",
    )

    # -----------------------------------------------------
    # TRADE-OFF
    # -----------------------------------------------------

    plot_tradeoff(
        summary_df
    )

    print()

    print(
        "=============================================="
    )

    print(
        "Finished."
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()