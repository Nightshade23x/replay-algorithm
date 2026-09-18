import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, t

from src.simulation.movielens_online_cf import (
    load_preferences,
    preference_to_click_probability,
)

from src.simulation.standardized_engine import (
    SimulationConfig,
    generate_warmup,
    generate_evaluation_randomness,
    simulate_algorithm,
)


# ---------------------------------------------------------
# FINAL ALGORITHMS
# ---------------------------------------------------------

ALGORITHMS = [
    {
        "name": "random",
        "algorithm": "random",
        "epsilon": 0.0,
    },
    {
        "name": "proportional_popularity",
        "algorithm": "proportional_popularity",
        "epsilon": 0.0,
    },
    {
        "name": "greedy_popularity",
        "algorithm": "greedy_popularity",
        "epsilon": 0.0,
    },
    {
        "name": "cf_e00",
        "algorithm": "cf",
        "epsilon": 0.00,
    },
    {
        "name": "cf_e10",
        "algorithm": "cf",
        "epsilon": 0.10,
    },
    {
        "name": "cf_e20",
        "algorithm": "cf",
        "epsilon": 0.20,
    },
]


NUM_WORLDS_FINAL = 50

BASE_SEED = 2026


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_comparison"
)


# ---------------------------------------------------------
# CONFIDENCE INTERVAL
# ---------------------------------------------------------

def mean_std_ci(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    se = (
        std
        / np.sqrt(
            n
        )
    )

    critical = t.ppf(
        0.975,
        df=n - 1,
    )

    margin = (
        critical
        * se
    )

    return (
        mean,
        std,
        mean - margin,
        mean + margin,
    )


# ---------------------------------------------------------
# PAIRWISE REPRODUCIBILITY
# ---------------------------------------------------------

def calculate_pairwise_metrics(
    algorithm_name,
    click_matrix,
    top_k,
):

    rows = []

    num_worlds = (
        click_matrix.shape[0]
    )

    for i in range(
        num_worlds
    ):

        for j in range(
            i + 1,
            num_worlds,
        ):

            clicks_a = (
                click_matrix[
                    i
                ]
            )

            clicks_b = (
                click_matrix[
                    j
                ]
            )

            correlation = (
                spearmanr(
                    clicks_a,
                    clicks_b,
                ).statistic
            )

            top_a = set(
                np.argsort(
                    clicks_a
                )[
                    -top_k:
                ]
            )

            top_b = set(
                np.argsort(
                    clicks_b
                )[
                    -top_k:
                ]
            )

            overlap = (
                len(
                    top_a
                    & top_b
                )
                / top_k
            )

            rows.append(
                {
                    "algorithm":
                        algorithm_name,

                    "world_a":
                        i,

                    "world_b":
                        j,

                    "rank_correlation":
                        correlation,

                    "top_k_overlap":
                        overlap,
                }
            )

    return rows


# ---------------------------------------------------------
# BUILD SUMMARY
# ---------------------------------------------------------

def build_summary(
    world_df,
    pairwise_df,
):

    metric_columns = [
        "ctr",
        "shown_relevance",
        "popularity_gini",
        "exposure_gini",
        "catalogue_coverage",
        "relevance_correlation",
        "top_k_relevance_recall",
        "total_clicks",
    ]

    rows = []

    for specification in (
        ALGORITHMS
    ):

        name = specification[
            "name"
        ]

        condition = world_df[
            world_df[
                "algorithm"
            ]
            == name
        ]

        pairwise_condition = (
            pairwise_df[
                pairwise_df[
                    "algorithm"
                ]
                == name
            ]
        )

        row = {
            "algorithm":
                name,

            "epsilon":
                specification[
                    "epsilon"
                ],

            "num_worlds":
                len(
                    condition
                ),

            "rank_correlation_mean":
                pairwise_condition[
                    "rank_correlation"
                ].mean(),

            "top_k_overlap_mean":
                pairwise_condition[
                    "top_k_overlap"
                ].mean(),
        }

        for metric in (
            metric_columns
        ):

            (
                mean,
                std,
                ci_low,
                ci_high,
            ) = mean_std_ci(
                condition[
                    metric
                ].to_numpy()
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

            row[
                f"{metric}_ci_low"
            ] = ci_low

            row[
                f"{metric}_ci_high"
            ] = ci_high

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a very small validation experiment "
            "before the full final experiment."
        ),
    )

    args = parser.parse_args()

    base_config = (
        SimulationConfig()
    )

    if args.smoke:

        config = replace(
            base_config,
            warmup_interactions=500,
            evaluation_interactions=2_000,
        )

        num_worlds = 3

        run_label = "SMOKE TEST"

    else:

        config = base_config

        num_worlds = (
            NUM_WORLDS_FINAL
        )

        run_label = (
            "FINAL STANDARDIZED COMPARISON"
        )

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

    underlying_item_relevance = (
        click_probabilities.mean(
            axis=0
        )
    )

    num_items = len(
        movie_ids
    )

    num_algorithms = len(
        ALGORITHMS
    )

    # -----------------------------------------------------
    # SAVE COMPLETE ITEM OUTCOMES
    # -----------------------------------------------------

    click_cube = np.zeros(
        (
            num_algorithms,
            num_worlds,
            num_items,
        ),
        dtype=np.int32,
    )

    exposure_cube = np.zeros(
        (
            num_algorithms,
            num_worlds,
            num_items,
        ),
        dtype=np.int32,
    )

    world_rows = []

    print(
        "============================================================"
    )

    print(
        run_label
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
        f"Worlds: "
        f"{num_worlds}"
    )

    print(
        f"Warm-up interactions: "
        f"{config.warmup_interactions:,}"
    )

    print(
        f"Evaluation interactions: "
        f"{config.evaluation_interactions:,}"
    )

    print(
        "Max impressions per user-item: "
        f"{config.max_impressions_per_user_item}"
    )

    print()

    # -----------------------------------------------------
    # RUN EACH PARALLEL WORLD
    # -----------------------------------------------------

    for world in range(
        num_worlds
    ):

        world_seed = (
            BASE_SEED
            + world
        )

        print(
            f"Preparing world "
            f"{world + 1}/"
            f"{num_worlds}"
        )

        # EXACT SAME RANDOM WARM-UP
        # FOR EVERY ALGORITHM.
        warmup = generate_warmup(
            click_probabilities,
            world_seed,
            config,
        )

        (
            evaluation_users,
            response_uniforms,
            exploration_uniforms,
        ) = generate_evaluation_randomness(
            world_seed,
            len(user_ids),
            config,
        )

        # -------------------------------------------------
        # RUN ALL ALGORITHMS FROM SAME WARM-UP
        # -------------------------------------------------

        for algorithm_index, specification in enumerate(
            ALGORITHMS
        ):

            name = specification[
                "name"
            ]

            print(
                f"  {name}",
                end="\r",
            )

            recommender_seed = (
                world_seed
                + 10_000_000
                + (
                    algorithm_index
                    * 100_000
                )
            )

            result = simulate_algorithm(
                algorithm=(
                    specification[
                        "algorithm"
                    ]
                ),
                epsilon=(
                    specification[
                        "epsilon"
                    ]
                ),
                click_probabilities=(
                    click_probabilities
                ),
                underlying_item_relevance=(
                    underlying_item_relevance
                ),
                warmup=warmup,
                evaluation_users=(
                    evaluation_users
                ),
                response_uniforms=(
                    response_uniforms
                ),
                exploration_uniforms=(
                    exploration_uniforms
                ),
                recommender_seed=(
                    recommender_seed
                ),
                config=config,
            )

            click_cube[
                algorithm_index,
                world,
            ] = result[
                "clicks"
            ]

            exposure_cube[
                algorithm_index,
                world,
            ] = result[
                "exposures"
            ]

            world_rows.append(
                {
                    "algorithm":
                        name,

                    "algorithm_type":
                        specification[
                            "algorithm"
                        ],

                    "epsilon":
                        specification[
                            "epsilon"
                        ],

                    "world":
                        world,

                    "seed":
                        world_seed,

                    "total_clicks":
                        result[
                            "total_clicks"
                        ],

                    "ctr":
                        result[
                            "ctr"
                        ],

                    "shown_relevance":
                        result[
                            "shown_relevance"
                        ],

                    "popularity_gini":
                        result[
                            "popularity_gini"
                        ],

                    "exposure_gini":
                        result[
                            "exposure_gini"
                        ],

                    "catalogue_coverage":
                        result[
                            "catalogue_coverage"
                        ],

                    "relevance_correlation":
                        result[
                            "relevance_correlation"
                        ],

                    "top_k_relevance_recall":
                        result[
                            "top_k_relevance_recall"
                        ],

                    "actual_exploration_rate":
                        result[
                            "actual_exploration_rate"
                        ],
                }
            )

        print(
            " " * 60,
            end="\r",
        )

    # -----------------------------------------------------
    # DATAFRAMES
    # -----------------------------------------------------

    world_df = pd.DataFrame(
        world_rows
    )

    pairwise_rows = []

    for algorithm_index, specification in enumerate(
        ALGORITHMS
    ):

        pairwise_rows.extend(
            calculate_pairwise_metrics(
                specification[
                    "name"
                ],
                click_cube[
                    algorithm_index
                ],
                config.top_k,
            )
        )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    summary_df = build_summary(
        world_df,
        pairwise_df,
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    suffix = (
        "_smoke"
        if args.smoke
        else ""
    )

    world_path = (
        OUTPUT_DIR
        / f"world_metrics{suffix}.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / f"pairwise_metrics{suffix}.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / f"summary{suffix}.csv"
    )

    item_path = (
        OUTPUT_DIR
        / f"item_outcomes{suffix}.npz"
    )

    config_path = (
        OUTPUT_DIR
        / f"config{suffix}.json"
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

    np.savez_compressed(
        item_path,
        algorithm_names=np.array(
            [
                specification[
                    "name"
                ]
                for specification
                in ALGORITHMS
            ]
        ),
        click_counts=click_cube,
        exposure_counts=exposure_cube,
        underlying_item_relevance=(
            underlying_item_relevance
        ),
        user_ids=user_ids,
        movie_ids=movie_ids,
    )

    config_data = {
        "num_worlds":
            num_worlds,

        "base_seed":
            BASE_SEED,

        "simulation":
            asdict(
                config
            ),

        "algorithms":
            ALGORITHMS,

        "response_probability_min":
            float(
                click_probabilities.min()
            ),

        "response_probability_max":
            float(
                click_probabilities.max()
            ),

        "response_probability_mean":
            float(
                click_probabilities.mean()
            ),
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_data,
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

    display_columns = [
        "algorithm",
        "rank_correlation_mean",
        "top_k_overlap_mean",
        "ctr_mean",
        "shown_relevance_mean",
        "popularity_gini_mean",
        "exposure_gini_mean",
        "catalogue_coverage_mean",
        "relevance_correlation_mean",
        "top_k_relevance_recall_mean",
    ]

    print(
        summary_df[
            display_columns
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
        item_path
    )

    print(
        config_path
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()