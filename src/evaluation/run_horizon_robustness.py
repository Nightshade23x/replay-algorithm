from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.recommenders.online_logistic_mf import (
    OnlineLogisticMF,
)

from src.simulation.movielens_online_cf import (
    load_preferences,
    preference_to_click_probability,
    random_recommender,
    gini,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 30

WARMUP_INTERACTIONS = 5_000

HORIZON_LEVELS = [
    10_000,
    25_000,
    50_000,
    75_000,
]

EPSILON = 0.10

CANDIDATE_POOL_SIZE = 50

TOP_K = 20

BASE_SEED = 2026
MODEL_SEED = 12345


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "horizon_robustness"
)


# ---------------------------------------------------------
# SIMULATE ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    evaluation_interactions,
    world_seed,
    click_probabilities,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    # Warm-up streams
    rng_warmup_users = np.random.default_rng(
        world_seed
    )

    rng_warmup_recommender = np.random.default_rng(
        world_seed + 1_000_000
    )

    rng_warmup_response = np.random.default_rng(
        world_seed + 2_000_000
    )

    # Evaluation streams are separate so every horizon
    # begins with the same post-warm-up random sequence.
    rng_eval_users = np.random.default_rng(
        world_seed + 3_000_000
    )

    rng_eval_recommender = np.random.default_rng(
        world_seed + 4_000_000
    )

    rng_eval_response = np.random.default_rng(
        world_seed + 5_000_000
    )

    seen = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=bool,
    )

    model = OnlineLogisticMF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=16,
        learning_rate=0.04,
        regularization=0.002,
        initial_click_rate=0.40,
        seed=MODEL_SEED,
    )

    # -----------------------------------------------------
    # FIXED RANDOM WARM-UP
    # -----------------------------------------------------

    for _ in range(
        WARMUP_INTERACTIONS
    ):

        user = rng_warmup_users.integers(
            num_users
        )

        item = random_recommender(
            rng_warmup_recommender,
            user,
            seen,
        )

        seen[
            user,
            item
        ] = True

        probability = click_probabilities[
            user,
            item
        ]

        outcome = int(
            rng_warmup_response.random()
            < probability
        )

        model.update(
            user,
            item,
            outcome,
        )

    # -----------------------------------------------------
    # EVALUATION PHASE
    # -----------------------------------------------------

    click_counts = np.zeros(
        num_items,
        dtype=int,
    )

    exposure_counts = np.zeros(
        num_items,
        dtype=int,
    )

    total_clicks = 0
    total_relevance = 0.0
    exploration_count = 0

    for _ in range(
        evaluation_interactions
    ):

        user = rng_eval_users.integers(
            num_users
        )

        explore = (
            rng_eval_recommender.random()
            < EPSILON
        )

        if explore:

            item = random_recommender(
                rng_eval_recommender,
                user,
                seen,
            )

            exploration_count += 1

        else:

            item = model.recommend(
                rng_eval_recommender,
                user,
                seen,
                candidate_pool_size=(
                    CANDIDATE_POOL_SIZE
                ),
            )

        exposure_counts[
            item
        ] += 1

        seen[
            user,
            item
        ] = True

        probability = click_probabilities[
            user,
            item
        ]

        total_relevance += probability

        outcome = int(
            rng_eval_response.random()
            < probability
        )

        if outcome == 1:

            click_counts[
                item
            ] += 1

            total_clicks += 1

        model.update(
            user,
            item,
            outcome,
        )

    return {
        "clicks":
            click_counts,

        "exposures":
            exposure_counts,

        "total_clicks":
            total_clicks,

        "ctr":
            (
                total_clicks
                / evaluation_interactions
            ),

        "shown_relevance":
            (
                total_relevance
                / evaluation_interactions
            ),

        "popularity_gini":
            gini(
                click_counts
            ),

        "exposure_gini":
            gini(
                exposure_counts
            ),

        "exploration_rate":
            (
                exploration_count
                / evaluation_interactions
            ),
    }


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def reproducibility_metrics(
    results,
):

    rank_correlations = []
    top_k_overlaps = []

    for i in range(
        len(results)
    ):

        for j in range(
            i + 1,
            len(results),
        ):

            clicks_a = results[i][
                "clicks"
            ]

            clicks_b = results[j][
                "clicks"
            ]

            correlation = spearmanr(
                clicks_a,
                clicks_b,
            ).statistic

            if not np.isnan(
                correlation
            ):

                rank_correlations.append(
                    correlation
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

            top_k_overlaps.append(
                len(
                    top_a & top_b
                )
                / TOP_K
            )

    return (
        np.mean(
            rank_correlations
        ),
        np.mean(
            top_k_overlaps
        ),
    )


# ---------------------------------------------------------
# MAIN
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
        "INTERACTION-HORIZON ROBUSTNESS EXPERIMENT"
    )

    print(
        "============================================================"
    )

    print(
        f"Users: {len(user_ids)}"
    )

    print(
        f"Items: {len(movie_ids)}"
    )

    print(
        f"Worlds per condition: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Warm-up interactions: "
        f"{WARMUP_INTERACTIONS:,}"
    )

    print(
        f"Exploration epsilon: "
        f"{EPSILON}"
    )

    print()

    world_rows = []
    summary_rows = []

    for horizon in (
        HORIZON_LEVELS
    ):

        print(
            f"Running evaluation horizon = "
            f"{horizon:,}"
        )

        results = []

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
                horizon,
                seed,
                click_probabilities,
            )

            results.append(
                result
            )

            world_rows.append(
                {
                    "horizon":
                        horizon,

                    "world":
                        world,

                    "seed":
                        seed,

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

                    "total_clicks":
                        result[
                            "total_clicks"
                        ],

                    "exploration_rate":
                        result[
                            "exploration_rate"
                        ],
                }
            )

        print(
            " " * 50,
            end="\r",
        )

        (
            rank_correlation,
            top_k_overlap,
        ) = reproducibility_metrics(
            results
        )

        summary_rows.append(
            {
                "horizon":
                    horizon,

                "rank_correlation":
                    rank_correlation,

                "top_k_overlap":
                    top_k_overlap,

                "ctr":
                    np.mean(
                        [
                            result[
                                "ctr"
                            ]
                            for result
                            in results
                        ]
                    ),

                "shown_relevance":
                    np.mean(
                        [
                            result[
                                "shown_relevance"
                            ]
                            for result
                            in results
                        ]
                    ),

                "popularity_gini":
                    np.mean(
                        [
                            result[
                                "popularity_gini"
                            ]
                            for result
                            in results
                        ]
                    ),

                "exposure_gini":
                    np.mean(
                        [
                            result[
                                "exposure_gini"
                            ]
                            for result
                            in results
                        ]
                    ),
            }
        )

    world_df = pd.DataFrame(
        world_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    world_path = (
        OUTPUT_DIR
        / "world_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    world_df.to_csv(
        world_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()

    print(
        "RESULTS"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        summary_df.to_string(
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
        summary_path
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()