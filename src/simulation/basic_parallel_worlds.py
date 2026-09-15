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
# HELPER FUNCTIONS
# ---------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gini(values):
    """
    Calculate the Gini coefficient.

    0 = perfectly equal popularity
    1 = extremely unequal popularity
    """

    values = np.asarray(values, dtype=float)

    if np.all(values == 0):
        return 0.0

    values = np.sort(values)

    n = len(values)

    cumulative = np.cumsum(values)

    return (
        (n + 1)
        - 2 * np.sum(cumulative) / cumulative[-1]
    ) / n


# ---------------------------------------------------------
# CREATE FIXED WORLD
# ---------------------------------------------------------

def create_true_preferences():
    """
    Create the underlying user-item preferences.

    These remain IDENTICAL in every simulated world.
    """

    rng = np.random.default_rng(BASE_SEED)

    user_factors = rng.normal(
        0,
        1,
        size=(NUM_USERS, LATENT_DIM),
    )

    item_factors = rng.normal(
        0,
        1,
        size=(NUM_ITEMS, LATENT_DIM),
    )

    raw_scores = (
        user_factors
        @ item_factors.T
        / np.sqrt(LATENT_DIM)
    )

    click_probabilities = sigmoid(
        raw_scores
    )

    return click_probabilities


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
        click_counts[available]
    )

    max_clicks = (
        available_clicks.max()
    )

    # There may be several equally popular items.
    # Random tie-breaking is one source of stochasticity.
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

        # -------------------------------------------------
        # EARLY RANDOM EXPOSURE
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

        else:

            if recommender_name == "random":

                item = random_recommender(
                    rng,
                    user,
                    seen,
                )

            elif recommender_name == "popularity":

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

        # Record exposure.
        exposure_counts[item] += 1

        seen[
            user,
            item
        ] = True

        # -------------------------------------------------
        # USER RESPONSE
        # -------------------------------------------------

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

            click_counts[item] += 1
            total_clicks += 1

    return {
        "click_counts":
            click_counts,

        "exposure_counts":
            exposure_counts,

        "total_clicks":
            total_clicks,

        "gini":
            gini(click_counts),
    }


# ---------------------------------------------------------
# COMPARE PARALLEL WORLDS
# ---------------------------------------------------------

def compare_worlds(
    results,
):

    correlations = []
    top_10_overlaps = []

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
                    top_a
                    & top_b
                )
                / 10
            )

            top_10_overlaps.append(
                overlap
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

        "mean_gini":
            np.mean(
                [
                    result["gini"]
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
# RUN EXPERIMENT
# ---------------------------------------------------------

def run_experiment(
    recommender_name,
    true_preferences,
):

    results = []

    for world in range(
        NUM_WORLDS
    ):

        result = simulate_world(
            recommender_name,
            world_seed=1000 + world,
            true_preferences=(
                true_preferences
            ),
        )

        results.append(
            result
        )

    return compare_worlds(
        results
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    true_preferences = (
        create_true_preferences()
    )

    print("==============================")
    print("PARALLEL WORLDS EXPERIMENT")
    print("==============================")

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
        f"Interactions per world: "
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
        )

        print(
            f"--- {recommender.upper()} ---"
        )

        print(
            "Mean rank correlation "
            f"between worlds: "
            f"{metrics['mean_rank_correlation']:.3f}"
        )

        print(
            "Mean top-10 overlap "
            f"between worlds: "
            f"{metrics['mean_top_10_overlap']:.3f}"
        )

        print(
            "Mean popularity Gini: "
            f"{metrics['mean_gini']:.3f}"
        )

        print(
            "Mean total clicks: "
            f"{metrics['mean_total_clicks']:.1f}"
        )

        print()

    print("==============================")


if __name__ == "__main__":
    main()