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

# Controls how strongly popularity influences recommendation.
#
# 0.0 = popularity ignored
# 1.0 = proportional to popularity
# >1  = increasingly strong popularity feedback
POPULARITY_ALPHA = 1.0

# Prevents items with zero clicks from receiving zero probability.
POPULARITY_PRIOR = 1.0


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
# LOAD MOVIELENS-DERIVED PREFERENCES
# ---------------------------------------------------------

def load_preferences():
    """
    Load the fixed MovieLens-derived preference matrix.

    Every simulated world uses exactly the same users,
    items and underlying preferences.
    """

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
# USER RESPONSE MODEL
# ---------------------------------------------------------

def preference_to_click_probability(
    preference_scores,
):
    """
    Convert MovieLens-derived preference scores into
    simulated positive-interaction probabilities.

    The preference scores are in [0, 1], but they are NOT
    directly treated as click probabilities.

    A logistic transformation is used instead.
    """

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
# GINI COEFFICIENT
# ---------------------------------------------------------

def gini(values):
    """
    Measure inequality.

    0 = perfectly equal distribution
    values closer to 1 = highly unequal distribution
    """

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
    """
    Recommend uniformly from items the user has not yet seen.
    """

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
# PROPORTIONAL POPULARITY RECOMMENDER
# ---------------------------------------------------------

def proportional_popularity_recommender(
    rng,
    user,
    seen,
    click_counts,
):
    """
    Recommend probabilistically according to popularity.

    Probability is proportional to:

        (clicks + prior) ^ alpha

    Unlike the greedy recommender, less-popular items can
    still receive exposure.
    """

    available = np.flatnonzero(
        ~seen[user]
    )

    if len(
        available
    ) == 0:

        available = np.arange(
            seen.shape[1]
        )

    weights = (
        click_counts[
            available
        ].astype(float)
        + POPULARITY_PRIOR
    ) ** POPULARITY_ALPHA

    probabilities = (
        weights
        / weights.sum()
    )

    return rng.choice(
        available,
        p=probabilities,
    )


# ---------------------------------------------------------
# GREEDY POPULARITY RECOMMENDER
# ---------------------------------------------------------

def greedy_popularity_recommender(
    rng,
    user,
    seen,
    click_counts,
):
    """
    Recommend one of the currently most popular available
    items.

    This is intentionally aggressive and acts as a
    stress-test baseline for strong feedback loops.
    """

    available = np.flatnonzero(
        ~seen[user]
    )

    if len(
        available
    ) == 0:

        available = np.arange(
            seen.shape[1]
        )

    available_clicks = (
        click_counts[
            available
        ]
    )

    maximum = (
        available_clicks.max()
    )

    candidates = available[
        available_clicks
        == maximum
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
    """
    Replay one simulated recommendation world.

    Users, items and underlying preferences remain fixed.

    The random seed changes between worlds, creating
    different early stochastic interactions.
    """

    rng = np.random.default_rng(
        world_seed
    )

    num_users, num_items = (
        click_probabilities.shape
    )

    click_counts = np.zeros(
        num_items,
        dtype=int,
    )

    exposure_counts = np.zeros(
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

        # -------------------------------------------------
        # INITIAL EXPLORATION
        # -------------------------------------------------

        if (
            step
            < INITIAL_RANDOM_INTERACTIONS
        ):

            item = random_recommender(
                rng,
                user,
                seen,
            )

        # -------------------------------------------------
        # RANDOM BASELINE
        # -------------------------------------------------

        elif (
            recommender_name
            == "random"
        ):

            item = random_recommender(
                rng,
                user,
                seen,
            )

        # -------------------------------------------------
        # PROPORTIONAL POPULARITY
        # -------------------------------------------------

        elif (
            recommender_name
            == "proportional_popularity"
        ):

            item = proportional_popularity_recommender(
                rng,
                user,
                seen,
                click_counts,
            )

        # -------------------------------------------------
        # GREEDY POPULARITY
        # -------------------------------------------------

        elif (
            recommender_name
            == "greedy_popularity"
        ):

            item = greedy_popularity_recommender(
                rng,
                user,
                seen,
                click_counts,
            )

        else:

            raise ValueError(
                f"Unknown recommender: "
                f"{recommender_name}"
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
        # SIMULATED USER RESPONSE
        # -------------------------------------------------

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

            click_counts[
                item
            ] += 1

            total_clicks += 1

    return {
        "clicks":
            click_counts,

        "exposures":
            exposure_counts,

        "total_clicks":
            total_clicks,

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
    underlying_item_relevance,
):
    """
    Evaluate:

    1. How similar outcomes are across parallel worlds.
    2. How closely popularity reflects underlying relevance.
    3. How unequal exposure and popularity become.
    """

    rank_correlations = []

    top_k_overlaps = []

    relevance_correlations = []

    true_top_k_recall = []

    # Items with the highest model-defined relevance.
    underlying_top_k = set(
        np.argsort(
            underlying_item_relevance
        )[-TOP_K:]
    )

    # -----------------------------------------------------
    # WORLD-TO-WORLD COMPARISON
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # UNDERLYING RELEVANCE VS OBSERVED POPULARITY
    # -----------------------------------------------------

    for result in results:

        clicks = result[
            "clicks"
        ]

        correlation = spearmanr(
            underlying_item_relevance,
            clicks,
        ).statistic

        if not np.isnan(
            correlation
        ):

            relevance_correlations.append(
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
                & underlying_top_k
            )
            / TOP_K
        )

        true_top_k_recall.append(
            recall
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    return {
        "rank_correlation":
            np.mean(
                rank_correlations
            ),

        "top_k_overlap":
            np.mean(
                top_k_overlaps
            ),

        "relevance_correlation":
            np.mean(
                relevance_correlations
            ),

        "top_k_relevance_recall":
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
# RUN ONE RECOMMENDER ACROSS ALL WORLDS
# ---------------------------------------------------------

def run_algorithm(
    recommender_name,
    click_probabilities,
    underlying_item_relevance,
):

    results = []

    for world in range(
        NUM_WORLDS
    ):

        world_seed = (
            BASE_SEED
            + world
        )

        result = simulate_world(
            recommender_name,
            world_seed,
            click_probabilities,
        )

        results.append(
            result
        )

    return evaluate_worlds(
        results,
        underlying_item_relevance,
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

    # Mean simulated positive-interaction probability
    # across all users.
    #
    # We call this "underlying item relevance" rather
    # than objective or true item quality.
    underlying_item_relevance = (
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

    print(
        f"Popularity alpha: "
        f"{POPULARITY_ALPHA}"
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

    recommenders = [
        "random",
        "proportional_popularity",
        "greedy_popularity",
    ]

    for recommender in recommenders:

        print(
            f"Running "
            f"{recommender}..."
        )

        metrics = run_algorithm(
            recommender,
            click_probabilities,
            underlying_item_relevance,
        )

        print()

        print(
            f"--- "
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
            "Underlying relevance vs popularity: "
            f"{metrics['relevance_correlation']:.3f}"
        )

        print(
            f"Underlying top-{TOP_K} recovered: "
            f"{metrics['top_k_relevance_recall']:.3f}"
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