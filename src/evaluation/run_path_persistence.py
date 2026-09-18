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
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 30

WARMUP_INTERACTIONS = 5_000

MAX_EVALUATION_INTERACTIONS = 75_000

# 75k itself is excluded because there would be
# no future period left to evaluate.
CHECKPOINTS = [
    5_000,
    10_000,
    25_000,
    50_000,
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
    / "path_persistence"
)


# ---------------------------------------------------------
# RUN ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    world_seed,
    click_probabilities,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    # -----------------------------------------------------
    # RANDOM STREAMS
    # -----------------------------------------------------

    rng_warmup_users = np.random.default_rng(
        world_seed
    )

    rng_warmup_rec = np.random.default_rng(
        world_seed + 1_000_000
    )

    rng_warmup_response = np.random.default_rng(
        world_seed + 2_000_000
    )

    rng_eval_users = np.random.default_rng(
        world_seed + 3_000_000
    )

    rng_eval_rec = np.random.default_rng(
        world_seed + 4_000_000
    )

    rng_eval_response = np.random.default_rng(
        world_seed + 5_000_000
    )

    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

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
    # RANDOM WARM-UP
    # -----------------------------------------------------

    for _ in range(
        WARMUP_INTERACTIONS
    ):

        user = rng_warmup_users.integers(
            num_users
        )

        item = random_recommender(
            rng_warmup_rec,
            user,
            seen,
        )

        seen[
            user,
            item
        ] = True

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

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
    # EVALUATION PERIOD
    # -----------------------------------------------------

    cumulative_clicks = np.zeros(
        num_items,
        dtype=int,
    )

    snapshots = {}

    for step in range(
        1,
        MAX_EVALUATION_INTERACTIONS + 1,
    ):

        user = rng_eval_users.integers(
            num_users
        )

        explore = (
            rng_eval_rec.random()
            < EPSILON
        )

        if explore:

            item = random_recommender(
                rng_eval_rec,
                user,
                seen,
            )

        else:

            item = model.recommend(
                rng_eval_rec,
                user,
                seen,
                candidate_pool_size=(
                    CANDIDATE_POOL_SIZE
                ),
            )

        seen[
            user,
            item
        ] = True

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

        outcome = int(
            rng_eval_response.random()
            < probability
        )

        if outcome == 1:

            cumulative_clicks[
                item
            ] += 1

        model.update(
            user,
            item,
            outcome,
        )

        if step in CHECKPOINTS:

            snapshots[
                step
            ] = cumulative_clicks.copy()

    # Save the final cumulative result separately.
    final_clicks = (
        cumulative_clicks.copy()
    )

    return (
        snapshots,
        final_clicks,
    )


# ---------------------------------------------------------
# COMPARE EARLY SUCCESS WITH FUTURE SUCCESS
# ---------------------------------------------------------

def compare_snapshots(
    snapshots,
    final_clicks,
):
    """
    Main persistence measure:

        early clicks
            VS
        clicks occurring AFTER the checkpoint

    This avoids mechanically including early clicks
    inside the outcome we are trying to predict.
    """

    rows = []

    final_top_k = set(
        np.argsort(
            final_clicks
        )[-TOP_K:]
    )

    for checkpoint in CHECKPOINTS:

        early_clicks = snapshots[
            checkpoint
        ]

        # Future-only success:
        #
        # clicks after checkpoint
        # =
        # final cumulative clicks - early cumulative clicks
        future_clicks = (
            final_clicks
            - early_clicks
        )

        # -------------------------------------------------
        # EARLY VS FUTURE-ONLY RANKING
        # -------------------------------------------------

        future_correlation = (
            spearmanr(
                early_clicks,
                future_clicks,
            ).statistic
        )

        early_top_k = set(
            np.argsort(
                early_clicks
            )[-TOP_K:]
        )

        future_top_k = set(
            np.argsort(
                future_clicks
            )[-TOP_K:]
        )

        future_top_k_overlap = (
            len(
                early_top_k
                & future_top_k
            )
            / TOP_K
        )

        # -------------------------------------------------
        # ALSO KEEP EARLY VS FINAL CUMULATIVE
        # AS A SECONDARY DESCRIPTIVE METRIC
        # -------------------------------------------------

        final_correlation = (
            spearmanr(
                early_clicks,
                final_clicks,
            ).statistic
        )

        final_top_k_overlap = (
            len(
                early_top_k
                & final_top_k
            )
            / TOP_K
        )

        rows.append(
            {
                "checkpoint":
                    checkpoint,

                "future_interactions":
                    (
                        MAX_EVALUATION_INTERACTIONS
                        - checkpoint
                    ),

                "early_vs_future_rank_correlation":
                    future_correlation,

                "early_vs_future_top_k_overlap":
                    future_top_k_overlap,

                "early_vs_final_rank_correlation":
                    final_correlation,

                "early_vs_final_top_k_overlap":
                    final_top_k_overlap,
            }
        )

    return rows


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
        "PATH PERSISTENCE EXPERIMENT"
    )

    print(
        "============================================================"
    )

    print(
        f"Worlds: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Warm-up: "
        f"{WARMUP_INTERACTIONS:,}"
    )

    print(
        f"Final evaluation horizon: "
        f"{MAX_EVALUATION_INTERACTIONS:,}"
    )

    print(
        f"Exploration epsilon: "
        f"{EPSILON}"
    )

    print()

    all_rows = []

    for world in range(
        NUM_WORLDS
    ):

        print(
            f"World "
            f"{world + 1}/"
            f"{NUM_WORLDS}",
            end="\r",
        )

        seed = (
            BASE_SEED
            + world
        )

        (
            snapshots,
            final_clicks,
        ) = simulate_world(
            seed,
            click_probabilities,
        )

        rows = compare_snapshots(
            snapshots,
            final_clicks,
        )

        for row in rows:

            row[
                "world"
            ] = world

            row[
                "seed"
            ] = seed

            all_rows.append(
                row
            )

    print(
        " " * 50,
        end="\r",
    )

    results_df = pd.DataFrame(
        all_rows
    )

    summary_df = (
        results_df
        .groupby(
            [
                "checkpoint",
                "future_interactions",
            ]
        )
        .agg(
            early_vs_future_rank_mean=(
                "early_vs_future_rank_correlation",
                "mean",
            ),

            early_vs_future_rank_std=(
                "early_vs_future_rank_correlation",
                "std",
            ),

            early_vs_future_top20_mean=(
                "early_vs_future_top_k_overlap",
                "mean",
            ),

            early_vs_future_top20_std=(
                "early_vs_future_top_k_overlap",
                "std",
            ),

            early_vs_final_rank_mean=(
                "early_vs_final_rank_correlation",
                "mean",
            ),

            early_vs_final_top20_mean=(
                "early_vs_final_top_k_overlap",
                "mean",
            ),
        )
        .reset_index()
    )

    results_path = (
        OUTPUT_DIR
        / "world_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    results_df.to_csv(
        results_path,
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
        results_path
    )

    print(
        summary_path
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()