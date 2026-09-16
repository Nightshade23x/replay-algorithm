from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.recommenders.online_logistic_mf import (
    OnlineLogisticMF,
)


# ---------------------------------------------------------
# DEVELOPMENT CONFIGURATION
# ---------------------------------------------------------

# Start small while validating the new recommender.
NUM_WORLDS = 10

INTERACTIONS_PER_WORLD = 30_000

WARMUP_INTERACTIONS = 5_000

TOP_K = 20

CANDIDATE_POOL_SIZE = 50

BASE_SEED = 2026

MODEL_SEED = 12345


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PREFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens_preferences.npz"
)


# ---------------------------------------------------------
# LOAD MOVIELENS WORLD
# ---------------------------------------------------------

def load_preferences():

    data = np.load(
        PREFERENCE_PATH
    )

    return (
        data[
            "preference_scores"
        ].astype(float),
        data["user_ids"],
        data["movie_ids"],
    )


# ---------------------------------------------------------
# RESPONSE MODEL
# ---------------------------------------------------------

def preference_to_click_probability(
    preference_scores,
):

    midpoint = 0.75

    strength = 6.0

    logits = (
        strength
        * (
            preference_scores
            - midpoint
        )
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -logits
            )
        )
    )


# ---------------------------------------------------------
# GINI
# ---------------------------------------------------------

def gini(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if np.all(
        values == 0
    ):
        return 0.0

    values = np.sort(
        values
    )

    n = len(
        values
    )

    cumulative = np.cumsum(
        values
    )

    return (
        (n + 1)
        - (
            2
            * np.sum(
                cumulative
            )
            / cumulative[-1]
        )
    ) / n


# ---------------------------------------------------------
# RANDOM RECOMMENDER
# ---------------------------------------------------------

def random_recommender(
    rng,
    user,
    seen,
):

    available = np.flatnonzero(
        ~seen[user]
    )

    if len(
        available
    ) == 0:

        available = np.arange(
            seen.shape[1]
        )

    return rng.choice(
        available
    )


# ---------------------------------------------------------
# SIMULATE ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    algorithm,
    world_seed,
    click_probabilities,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    # Separate random streams make the experiment cleaner.
    #
    # The user-arrival sequence and response randomness are
    # matched across algorithms for the same world seed.
    rng_users = np.random.default_rng(
        world_seed
    )

    rng_recommender = np.random.default_rng(
        world_seed
        + 1_000_000
    )

    rng_response = np.random.default_rng(
        world_seed
        + 2_000_000
    )

    seen = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=bool,
    )

    click_counts = np.zeros(
        num_items,
        dtype=int,
    )

    exposure_counts = np.zeros(
        num_items,
        dtype=int,
    )

    total_clicks = 0

    # This uses hidden simulator information ONLY for
    # evaluation, never for selecting recommendations.
    total_oracle_probability = 0.0

    if algorithm == "online_cf":

        model = OnlineLogisticMF(
            num_users=num_users,
            num_items=num_items,
            latent_dim=16,
            learning_rate=0.04,
            regularization=0.002,
            initial_click_rate=0.40,
            seed=MODEL_SEED,
        )

    else:

        model = None

    for step in range(
        INTERACTIONS_PER_WORLD
    ):

        user = rng_users.integers(
            num_users
        )

        # -------------------------------------------------
        # WARM-UP
        # -------------------------------------------------
        #
        # Both algorithms receive random exposure during
        # this period.
        #
        # The CF model learns from these interactions.

        if (
            step
            < WARMUP_INTERACTIONS
        ):

            item = random_recommender(
                rng_recommender,
                user,
                seen,
            )

        # -------------------------------------------------
        # RANDOM BASELINE
        # -------------------------------------------------

        elif (
            algorithm
            == "random"
        ):

            item = random_recommender(
                rng_recommender,
                user,
                seen,
            )

        # -------------------------------------------------
        # ONLINE COLLABORATIVE FILTERING
        # -------------------------------------------------

        elif (
            algorithm
            == "online_cf"
        ):

            item = model.recommend(
                rng_recommender,
                user,
                seen,
                candidate_pool_size=(
                    CANDIDATE_POOL_SIZE
                ),
            )

        else:

            raise ValueError(
                f"Unknown algorithm: "
                f"{algorithm}"
            )

        # -------------------------------------------------
        # RECORD EXPOSURE
        # -------------------------------------------------

        exposure_counts[
            item
        ] += 1

        seen[
            user,
            item
        ] = True

        # -------------------------------------------------
        # HIDDEN SIMULATOR RESPONSE PROBABILITY
        # -------------------------------------------------

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

        total_oracle_probability += (
            probability
        )

        clicked = (
            rng_response.random()
            < probability
        )

        outcome = int(
            clicked
        )

        if clicked:

            click_counts[
                item
            ] += 1

            total_clicks += 1

        # -------------------------------------------------
        # ONLINE LEARNING
        # -------------------------------------------------
        #
        # The model sees only:
        #
        # user
        # exposed item
        # click / no click

        if model is not None:

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
                / INTERACTIONS_PER_WORLD
            ),

        "mean_oracle_probability":
            (
                total_oracle_probability
                / INTERACTIONS_PER_WORLD
            ),

        "click_gini":
            gini(
                click_counts
            ),

        "exposure_gini":
            gini(
                exposure_counts
            ),
    }


# ---------------------------------------------------------
# EVALUATE PARALLEL WORLDS
# ---------------------------------------------------------

def evaluate_worlds(
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

            clicks_a = (
                results[i][
                    "clicks"
                ]
            )

            clicks_b = (
                results[j][
                    "clicks"
                ]
            )

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
                    top_a
                    & top_b
                )
                / TOP_K
            )

    return {
        "rank_correlation":
            np.mean(
                rank_correlations
            ),

        "top_k_overlap":
            np.mean(
                top_k_overlaps
            ),

        "click_gini":
            np.mean(
                [
                    result[
                        "click_gini"
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

        "mean_clicks":
            np.mean(
                [
                    result[
                        "total_clicks"
                    ]
                    for result
                    in results
                ]
            ),

        "mean_ctr":
            np.mean(
                [
                    result[
                        "ctr"
                    ]
                    for result
                    in results
                ]
            ),

        "mean_oracle_probability":
            np.mean(
                [
                    result[
                        "mean_oracle_probability"
                    ]
                    for result
                    in results
                ]
            ),
    }


# ---------------------------------------------------------
# RUN ALGORITHM
# ---------------------------------------------------------

def run_algorithm(
    algorithm,
    click_probabilities,
):

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

        result = simulate_world(
            algorithm,
            BASE_SEED + world,
            click_probabilities,
        )

        results.append(
            result
        )

    print(
        " " * 40,
        end="\r",
    )

    return evaluate_worlds(
        results
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

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
        "=============================================="
    )

    print(
        "MOVIELENS ONLINE COLLABORATIVE FILTERING"
    )

    print(
        "=============================================="
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
        f"{NUM_WORLDS}"
    )

    print(
        f"Interactions per world: "
        f"{INTERACTIONS_PER_WORLD:,}"
    )

    print(
        f"Random warm-up interactions: "
        f"{WARMUP_INTERACTIONS:,}"
    )

    print(
        f"CF candidate pool: "
        f"{CANDIDATE_POOL_SIZE}"
    )

    print()

    for algorithm in [
        "random",
        "online_cf",
    ]:

        print(
            f"Running "
            f"{algorithm}..."
        )

        metrics = run_algorithm(
            algorithm,
            click_probabilities,
        )

        print(
            f"\n--- "
            f"{algorithm.upper()} "
            f"---"
        )

        print(
            "World-to-world rank correlation: "
            f"{metrics['rank_correlation']:.3f}"
        )

        print(
            f"World-to-world top-{TOP_K} overlap: "
            f"{metrics['top_k_overlap']:.3f}"
        )

        print(
            "Popularity Gini: "
            f"{metrics['click_gini']:.3f}"
        )

        print(
            "Exposure Gini: "
            f"{metrics['exposure_gini']:.3f}"
        )

        print(
            "Mean total clicks: "
            f"{metrics['mean_clicks']:.1f}"
        )

        print(
            "Mean click-through rate: "
            f"{metrics['mean_ctr']:.3f}"
        )

        print(
            "Mean underlying response probability "
            "of shown items: "
            f"{metrics['mean_oracle_probability']:.3f}"
        )

        print()

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()