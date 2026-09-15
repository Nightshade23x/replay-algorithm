import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_USERS = 100
NUM_ITEMS = 100
LATENT_DIM = 5

INTERACTIONS_PER_WORLD = 5000
NUM_WORLDS = 50

INITIAL_RANDOM_INTERACTIONS = 200

BASE_SEED = 42


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gini(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if np.all(values == 0):
        return 0.0

    values = np.sort(values)

    n = len(values)

    cumulative = np.cumsum(values)

    return (
        (n + 1)
        - 2 * np.sum(cumulative)
        / cumulative[-1]
    ) / n


# ---------------------------------------------------------
# FIXED UNDERLYING WORLD
# ---------------------------------------------------------

def create_true_preferences():
    """
    Create persistent user preferences and item quality.

    These are identical in every parallel world.

    Each user's probability of clicking an item depends on:

        global item quality
            +
        personal user-item affinity

    Therefore genuinely strong items exist, while users
    can still have different tastes.
    """

    rng = np.random.default_rng(
        BASE_SEED
    )

    # Personal taste
    user_factors = rng.normal(
        0,
        0.7,
        size=(
            NUM_USERS,
            LATENT_DIM,
        ),
    )

    item_factors = rng.normal(
        0,
        0.7,
        size=(
            NUM_ITEMS,
            LATENT_DIM,
        ),
    )

    # Persistent intrinsic quality of each item.
    item_quality = rng.normal(
        0,
        0.8,
        size=NUM_ITEMS,
    )

    personal_affinity = (
        user_factors
        @ item_factors.T
        / np.sqrt(LATENT_DIM)
    )

    raw_scores = (
        personal_affinity
        + item_quality[None, :]
    )

    click_probabilities = sigmoid(
        raw_scores
    )

    # Operational "true quality":
    # probability of being clicked by an average user.
    true_item_quality = (
        click_probabilities.mean(
            axis=0
        )
    )

    return (
        click_probabilities,
        true_item_quality,
    )


# ---------------------------------------------------------
# RECOMMENDERS
# ---------------------------------------------------------

def random_recommender(
    rng,
    user,
    seen,
):

    available = np.where(
        ~seen[user]
    )[0]

    if len(available) == 0:

        available = np.arange(
            NUM_ITEMS
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

    available = np.where(
        ~seen[user]
    )[0]

    if len(available) == 0:

        available = np.arange(
            NUM_ITEMS
        )

    available_clicks = (
        click_counts[
            available
        ]
    )

    max_clicks = (
        available_clicks.max()
    )

    candidates = available[
        available_clicks
        == max_clicks
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
    true_preferences,
):

    rng = np.random.default_rng(
        world_seed
    )

    click_counts = np.zeros(
        NUM_ITEMS,
        dtype=int,
    )

    exposure_counts = np.zeros(
        NUM_ITEMS,
        dtype=int,
    )

    seen = np.zeros(
        (
            NUM_USERS,
            NUM_ITEMS,
        ),
        dtype=bool,
    )

    total_clicks = 0

    for step in range(
        INTERACTIONS_PER_WORLD
    ):

        user = rng.integers(
            NUM_USERS
        )

        # ---------------------------------------------
        # INITIAL RANDOM EXPOSURE
        # ---------------------------------------------

        if (
            step
            < INITIAL_RANDOM_INTERACTIONS
        ):

            item = random_recommender(
                rng,
                user,
                seen,
            )

        else:

            if (
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
                    click_counts,
                )

            else:

                raise ValueError(
                    f"Unknown recommender: "
                    f"{recommender_name}"
                )

        # ---------------------------------------------
        # EXPOSURE
        # ---------------------------------------------

        exposure_counts[
            item
        ] += 1

        seen[
            user,
            item
        ] = True

        # ---------------------------------------------
        # USER RESPONSE
        # ---------------------------------------------

        probability = (
            true_preferences[
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
        "click_counts":
            click_counts,

        "exposure_counts":
            exposure_counts,

        "total_clicks":
            total_clicks,

        "popularity_gini":
            gini(
                click_counts
            ),

        "exposure_gini":
            gini(
                exposure_counts
            ),
    }


# ---------------------------------------------------------
# COMPARE PARALLEL WORLDS
# ---------------------------------------------------------

def compare_worlds(
    results,
    true_item_quality,
):

    correlations = []
    top_10_overlaps = []

    quality_correlations = []
    true_top_10_recall = []

    true_top_10 = set(
        np.argsort(
            true_item_quality
        )[-10:]
    )

    # ---------------------------------------------
    # WORLD-TO-WORLD COMPARISON
    # ---------------------------------------------

    for i in range(
        len(results)
    ):

        for j in range(
            i + 1,
            len(results),
        ):

            clicks_a = (
                results[i][
                    "click_counts"
                ]
            )

            clicks_b = (
                results[j][
                    "click_counts"
                ]
            )

            correlation = spearmanr(
                clicks_a,
                clicks_b,
            ).statistic

            if not np.isnan(
                correlation
            ):

                correlations.append(
                    correlation
                )

            top_a = set(
                np.argsort(
                    clicks_a
                )[-10:]
            )

            top_b = set(
                np.argsort(
                    clicks_b
                )[-10:]
            )

            overlap = (
                len(
                    top_a & top_b
                )
                / 10
            )

            top_10_overlaps.append(
                overlap
            )

    # ---------------------------------------------
    # TRUE QUALITY VS OBSERVED SUCCESS
    # ---------------------------------------------

    for result in results:

        clicks = result[
            "click_counts"
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

        observed_top_10 = set(
            np.argsort(
                clicks
            )[-10:]
        )

        recall = (
            len(
                observed_top_10
                & true_top_10
            )
            / 10
        )

        true_top_10_recall.append(
            recall
        )

    return {
        "mean_rank_correlation":
            np.mean(
                correlations
            ),

        "mean_top_10_overlap":
            np.mean(
                top_10_overlaps
            ),

        "mean_popularity_gini":
            np.mean(
                [
                    result[
                        "popularity_gini"
                    ]
                    for result
                    in results
                ]
            ),

        "mean_exposure_gini":
            np.mean(
                [
                    result[
                        "exposure_gini"
                    ]
                    for result
                    in results
                ]
            ),

        "mean_quality_correlation":
            np.mean(
                quality_correlations
            ),

        "mean_true_top_10_recall":
            np.mean(
                true_top_10_recall
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
# RUN EXPERIMENT
# ---------------------------------------------------------

def run_experiment(
    recommender_name,
    true_preferences,
    true_item_quality,
):

    results = []

    for world in range(
        NUM_WORLDS
    ):

        result = simulate_world(
            recommender_name,
            world_seed=(
                1000 + world
            ),
            true_preferences=(
                true_preferences
            ),
        )

        results.append(
            result
        )

    return compare_worlds(
        results,
        true_item_quality,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    (
        true_preferences,
        true_item_quality,
    ) = create_true_preferences()

    print(
        "=============================="
    )
    print(
        "PARALLEL WORLDS EXPERIMENT V2"
    )
    print(
        "=============================="
    )

    print(
        f"Users: {NUM_USERS}"
    )

    print(
        f"Items: {NUM_ITEMS}"
    )

    print(
        f"Worlds: {NUM_WORLDS}"
    )

    print(
        "Interactions per world: "
        f"{INTERACTIONS_PER_WORLD:,}"
    )

    print()

    for recommender in [
        "random",
        "popularity",
    ]:

        metrics = run_experiment(
            recommender,
            true_preferences,
            true_item_quality,
        )

        print(
            f"--- "
            f"{recommender.upper()} "
            f"---"
        )

        print(
            "World-to-world rank "
            "correlation: "
            f"{metrics['mean_rank_correlation']:.3f}"
        )

        print(
            "World-to-world top-10 "
            "overlap: "
            f"{metrics['mean_top_10_overlap']:.3f}"
        )

        print(
            "Popularity Gini: "
            f"{metrics['mean_popularity_gini']:.3f}"
        )

        print(
            "Exposure Gini: "
            f"{metrics['mean_exposure_gini']:.3f}"
        )

        print(
            "True quality vs popularity "
            "correlation: "
            f"{metrics['mean_quality_correlation']:.3f}"
        )

        print(
            "True top-10 recovered: "
            f"{metrics['mean_true_top_10_recall']:.3f}"
        )

        print(
            "Mean total clicks: "
            f"{metrics['mean_total_clicks']:.1f}"
        )

        print()

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()