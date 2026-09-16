from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 50

INTERACTIONS_PER_WORLD = 50_000

INITIAL_RANDOM_INTERACTIONS = 2_000

TOP_K = 20

BASE_SEED = 2026


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
# LOAD MOVIELENS-DERIVED WORLD
# ---------------------------------------------------------

def load_preferences():

    data = np.load(
        PREFERENCE_PATH
    )

    preference_scores = data[
        "preference_scores"
    ].astype(float)

    user_ids = data[
        "user_ids"
    ]

    movie_ids = data[
        "movie_ids"
    ]

    return (
        preference_scores,
        user_ids,
        movie_ids,
    )


# ---------------------------------------------------------
# RESPONSE MODEL
# ---------------------------------------------------------

def preference_to_click_probability(
    preference_scores,
):
    """
    Convert MovieLens-derived preference scores into
    simulated click probabilities.

    preference_scores are in [0, 1].

    IMPORTANT:
    These scores are not themselves click probabilities.

    We map them through a logistic response model so that
    highly preferred items are substantially more likely
    to receive positive interaction.
    """

    # Corresponds roughly to a predicted MovieLens
    # rating of 4 out of 5.
    midpoint = 0.75

    strength = 6.0

    logits = (
        strength
        * (
            preference_scores
            - midpoint
        )
    )

    probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -logits
            )
        )
    )

    return probabilities


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
# RECOMMENDERS
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


def popularity_recommender(
    rng,
    user,
    seen,
    click_counts,
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

    counts = click_counts[
        available
    ]

    maximum = counts.max()

    candidates = available[
        counts == maximum
    ]

    return rng.choice(
        candidates
    )


# ---------------------------------------------------------
# SIMULATE ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    recommender_name,
    world_seed,
    click_probabilities,
):

    rng = np.random.default_rng(
        world_seed
    )

    num_users, num_items = (
        click_probabilities.shape
    )

    clicks = np.zeros(
        num_items,
        dtype=int,
    )

    exposures = np.zeros(
        num_items,
        dtype=int,
    )

    seen = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=bool,
    )

    total_clicks = 0

    for step in range(
        INTERACTIONS_PER_WORLD
    ):

        user = rng.integers(
            num_users
        )

        # Initial exploration period.
        if (
            step
            < INITIAL_RANDOM_INTERACTIONS
        ):

            item = random_recommender(
                rng,
                user,
                seen,
            )

        elif (
            recommender_name
            == "random"
        ):

            item = random_recommender(
                rng,
                user,
                seen,
            )

        elif (
            recommender_name
            == "popularity"
        ):

            item = popularity_recommender(
                rng,
                user,
                seen,
                clicks,
            )

        else:

            raise ValueError(
                f"Unknown recommender: "
                f"{recommender_name}"
            )

        exposures[
            item
        ] += 1

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

        clicked = (
            rng.random()
            < probability
        )

        if clicked:

            clicks[
                item
            ] += 1

            total_clicks += 1

    return {
        "clicks":
            clicks,

        "exposures":
            exposures,

        "total_clicks":
            total_clicks,

        "click_gini":
            gini(
                clicks
            ),

        "exposure_gini":
            gini(
                exposures
            ),
    }


# ---------------------------------------------------------
# COMPARE WORLDS
# ---------------------------------------------------------

def evaluate_worlds(
    results,
    true_item_quality,
):

    rank_correlations = []
    top_k_overlaps = []

    quality_correlations = []

    true_top_k = set(
        np.argsort(
            true_item_quality
        )[-TOP_K:]
    )

    true_top_k_recall = []

    # Pairwise world comparisons.
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

            overlap = (
                len(
                    top_a
                    & top_b
                )
                / TOP_K
            )

            top_k_overlaps.append(
                overlap
            )

    # Compare success with underlying quality.
    for result in results:

        clicks = result[
            "clicks"
        ]

        correlation = spearmanr(
            true_item_quality,
            clicks,
        ).statistic

        if not np.isnan(
            correlation
        ):

            quality_correlations.append(
                correlation
            )

        observed_top_k = set(
            np.argsort(
                clicks
            )[-TOP_K:]
        )

        recall = (
            len(
                observed_top_k
                & true_top_k
            )
            / TOP_K
        )

        true_top_k_recall.append(
            recall
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

        "quality_correlation":
            np.mean(
                quality_correlations
            ),

        "true_top_k_recall":
            np.mean(
                true_top_k_recall
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

        "mean_total_clicks":
            np.mean(
                [
                    result[
                        "total_clicks"
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
    recommender_name,
    click_probabilities,
    true_item_quality,
):

    results = []

    for world in range(
        NUM_WORLDS
    ):

        result = simulate_world(
            recommender_name,
            BASE_SEED + world,
            click_probabilities,
        )

        results.append(
            result
        )

    return evaluate_worlds(
        results,
        true_item_quality,
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

    # "True item quality" in this simulated world:
    # mean positive-interaction probability across users.
    true_item_quality = (
        click_probabilities.mean(
            axis=0
        )
    )

    print(
        "=============================================="
    )

    print(
        "MOVIELENS PARALLEL WORLDS"
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
        f"Parallel worlds: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Interactions per world: "
        f"{INTERACTIONS_PER_WORLD:,}"
    )

    print(
        f"Initial random interactions: "
        f"{INITIAL_RANDOM_INTERACTIONS:,}"
    )

    print()

    print(
        "Simulated response probabilities:"
    )

    print(
        f"Minimum: "
        f"{click_probabilities.min():.3f}"
    )

    print(
        f"Maximum: "
        f"{click_probabilities.max():.3f}"
    )

    print(
        f"Mean: "
        f"{click_probabilities.mean():.3f}"
    )

    print()

    for recommender in [
        "random",
        "popularity",
    ]:

        print(
            f"Running "
            f"{recommender}..."
        )

        metrics = run_algorithm(
            recommender,
            click_probabilities,
            true_item_quality,
        )

        print(
            f"\n--- "
            f"{recommender.upper()} "
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
            "True quality vs popularity: "
            f"{metrics['quality_correlation']:.3f}"
        )

        print(
            f"True top-{TOP_K} recovered: "
            f"{metrics['true_top_k_recall']:.3f}"
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
            f"{metrics['mean_total_clicks']:.1f}"
        )

        print()

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()