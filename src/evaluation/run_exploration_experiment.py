import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, t

from src.simulation.movielens_online_cf import (
    load_preferences,
    preference_to_click_probability,
)

from src.simulation.movielens_exploration_cf import (
    simulate_world,
    INTERACTIONS_PER_WORLD,
    WARMUP_INTERACTIONS,
    CANDIDATE_POOL_SIZE,
    BASE_SEED,
)


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 50

TOP_K = 20

EPSILON_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "exploration_cf"
)


# ---------------------------------------------------------
# CONFIDENCE INTERVAL
# ---------------------------------------------------------

def confidence_interval_95(
    values,
):
    """
    95% t confidence interval for independent
    per-world measurements.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    standard_error = (
        std
        / np.sqrt(n)
    )

    critical_value = t.ppf(
        0.975,
        df=n - 1,
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        mean,
        std,
        mean - margin,
        mean + margin,
    )


# ---------------------------------------------------------
# PAIRWISE WORLD COMPARISONS
# ---------------------------------------------------------

def calculate_pairwise_metrics(
    epsilon,
    world_results,
):

    rows = []

    for i in range(
        len(world_results)
    ):

        for j in range(
            i + 1,
            len(world_results),
        ):

            clicks_a = (
                world_results[i][
                    "clicks"
                ]
            )

            clicks_b = (
                world_results[j][
                    "clicks"
                ]
            )

            rank_correlation = (
                spearmanr(
                    clicks_a,
                    clicks_b,
                ).statistic
            )

            top_a = set(
                np.argsort(
                    clicks_a
                )[-TOP_K:]
            )

            top_b = set(
                np.argsort(
                    clicks_b
                )[-TOP_K:]
            )

            top_k_overlap = (
                len(
                    top_a
                    & top_b
                )
                / TOP_K
            )

            rows.append(
                {
                    "epsilon":
                        epsilon,

                    "world_a":
                        i,

                    "world_b":
                        j,

                    "rank_correlation":
                        rank_correlation,

                    "top_k_overlap":
                        top_k_overlap,
                }
            )

    return rows


# ---------------------------------------------------------
# SUMMARISE PER-WORLD METRICS
# ---------------------------------------------------------

def build_summary(
    world_df,
    pairwise_df,
):

    summary_rows = []

    metrics = {
        "ctr":
            "ctr",

        "shown_relevance":
            "mean_oracle_probability",

        "popularity_gini":
            "click_gini",

        "exposure_gini":
            "exposure_gini",

        "total_clicks":
            "total_clicks",

        "exploration_rate":
            "actual_exploration_rate",
    }

    for epsilon in (
        EPSILON_LEVELS
    ):

        condition = world_df[
            world_df[
                "epsilon"
            ]
            == epsilon
        ]

        pairwise_condition = (
            pairwise_df[
                pairwise_df[
                    "epsilon"
                ]
                == epsilon
            ]
        )

        row = {
            "epsilon":
                epsilon,

            "num_worlds":
                len(
                    condition
                ),

            # Pairwise comparisons are saved for
            # descriptive reproducibility analysis.
            #
            # We do NOT calculate a naive CI over these
            # pairs because the pairs are not independent.
            "rank_correlation_mean":
                pairwise_condition[
                    "rank_correlation"
                ].mean(),

            "top_k_overlap_mean":
                pairwise_condition[
                    "top_k_overlap"
                ].mean(),
        }

        for output_name, column in (
            metrics.items()
        ):

            (
                mean,
                std,
                ci_low,
                ci_high,
            ) = confidence_interval_95(
                condition[
                    column
                ].to_numpy()
            )

            row[
                f"{output_name}_mean"
            ] = mean

            row[
                f"{output_name}_std"
            ] = std

            row[
                f"{output_name}_ci_low"
            ] = ci_low

            row[
                f"{output_name}_ci_high"
            ] = ci_high

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        preference_scores,
        user_ids,
        movie_ids,
    ) = load_preferences()

    click_probabilities = (
        preference_to_click_probability(
            preference_scores
        )
    )

    print(
        "============================================================"
    )

    print(
        "THESIS EXPLORATION EXPERIMENT"
    )

    print(
        "============================================================"
    )

    print(
        f"Users: "
        f"{len(user_ids)}"
    )

    print(
        f"Items: "
        f"{len(movie_ids)}"
    )

    print(
        f"Worlds per condition: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Interactions per world: "
        f"{INTERACTIONS_PER_WORLD:,}"
    )

    print()

    world_rows = []

    pairwise_rows = []

    # -----------------------------------------------------
    # RUN EVERY EPSILON CONDITION
    # -----------------------------------------------------

    for epsilon in (
        EPSILON_LEVELS
    ):

        print(
            f"Running epsilon = "
            f"{epsilon:.2f}"
        )

        condition_results = []

        for world in range(
            NUM_WORLDS
        ):

            print(
                f"  World "
                f"{world + 1}/"
                f"{NUM_WORLDS}",
                end="\r",
            )

            seed = (
                BASE_SEED
                + world
            )

            result = simulate_world(
                epsilon,
                seed,
                click_probabilities,
            )

            condition_results.append(
                result
            )

            world_rows.append(
                {
                    "epsilon":
                        epsilon,

                    "world":
                        world,

                    "seed":
                        seed,

                    "total_clicks":
                        result[
                            "total_clicks"
                        ],

                    "ctr":
                        result[
                            "ctr"
                        ],

                    "mean_oracle_probability":
                        result[
                            "mean_oracle_probability"
                        ],

                    "click_gini":
                        result[
                            "click_gini"
                        ],

                    "exposure_gini":
                        result[
                            "exposure_gini"
                        ],

                    "actual_exploration_rate":
                        result[
                            "actual_exploration_rate"
                        ],
                }
            )

        print(
            " " * 50,
            end="\r",
        )

        pairwise_rows.extend(
            calculate_pairwise_metrics(
                epsilon,
                condition_results,
            )
        )

    # -----------------------------------------------------
    # SAVE RAW RESULTS
    # -----------------------------------------------------

    world_df = pd.DataFrame(
        world_rows
    )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    summary_df = build_summary(
        world_df,
        pairwise_df,
    )

    world_path = (
        OUTPUT_DIR
        / "world_metrics.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / "pairwise_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    world_df.to_csv(
        world_path,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE CONFIGURATION
    # -----------------------------------------------------

    config = {
        "num_worlds":
            NUM_WORLDS,

        "num_users":
            int(
                len(user_ids)
            ),

        "num_items":
            int(
                len(movie_ids)
            ),

        "interactions_per_world":
            INTERACTIONS_PER_WORLD,

        "warmup_interactions":
            WARMUP_INTERACTIONS,

        "candidate_pool_size":
            CANDIDATE_POOL_SIZE,

        "base_seed":
            BASE_SEED,

        "top_k":
            TOP_K,

        "epsilon_levels":
            EPSILON_LEVELS,
    }

    config_path = (
        OUTPUT_DIR
        / "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # TERMINAL SUMMARY
    # -----------------------------------------------------

    print()

    print(
        "RESULTS"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        summary_df[
            [
                "epsilon",
                "rank_correlation_mean",
                "top_k_overlap_mean",
                "popularity_gini_mean",
                "exposure_gini_mean",
                "ctr_mean",
                "shown_relevance_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        world_path
    )

    print(
        pairwise_path
    )

    print(
        summary_path
    )

    print(
        config_path
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()