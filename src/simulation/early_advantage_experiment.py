import numpy as np


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 1000

FUTURE_INTERACTIONS = 5000

# Both items begin with this many historical clicks.
BASE_SEED_CLICKS = 100

# Item A receives this percentage more historical clicks
# than Item B.
ADVANTAGE_LEVELS = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.20,
]

# The two items are genuinely identical.
CLICK_PROBABILITY = 0.50

# Strength of popularity feedback.
#
# alpha = 0 -> popularity has no effect
# alpha = 1 -> recommendation probability is proportional
#              to current popularity
POPULARITY_ALPHA = 1.0

BASE_RANDOM_SEED = 2026

ITEM_A = 0
ITEM_B = 1


# ---------------------------------------------------------
# RECOMMENDERS
# ---------------------------------------------------------

def random_recommender(rng):
    """
    Give A and B equal probability of being recommended.
    """

    return rng.integers(0, 2)


def popularity_recommender(
    rng,
    click_counts,
):
    """
    Recommend items according to their current popularity.

    The +1 prevents zero-count items from having zero
    recommendation probability.
    """

    weights = (
        click_counts.astype(float)
        + 1.0
    ) ** POPULARITY_ALPHA

    probabilities = (
        weights
        / weights.sum()
    )

    return rng.choice(
        [ITEM_A, ITEM_B],
        p=probabilities,
    )


# ---------------------------------------------------------
# SIMULATE ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    recommender_name,
    advantage,
    world_seed,
):
    """
    Simulate one replay of the same world.

    Item A and Item B have exactly the same true quality.

    The only deliberate difference is that Item A begins
    with a small historical popularity advantage.
    """

    rng = np.random.default_rng(
        world_seed
    )

    seed_clicks_b = (
        BASE_SEED_CLICKS
    )

    seed_clicks_a = round(
        BASE_SEED_CLICKS
        * (1 + advantage)
    )

    # Historical popularity seen by the recommender.
    click_counts = np.array(
        [
            seed_clicks_a,
            seed_clicks_b,
        ],
        dtype=int,
    )

    # Keep future outcomes separate from the seed.
    #
    # This is important because we want to know whether
    # the initial advantage causes ADDITIONAL future success.
    future_clicks = np.zeros(
        2,
        dtype=int,
    )

    future_exposures = np.zeros(
        2,
        dtype=int,
    )

    for _ in range(
        FUTURE_INTERACTIONS
    ):

        if recommender_name == "random":

            item = random_recommender(
                rng
            )

        elif recommender_name == "popularity":

            item = popularity_recommender(
                rng,
                click_counts,
            )

        else:

            raise ValueError(
                f"Unknown recommender: "
                f"{recommender_name}"
            )

        future_exposures[
            item
        ] += 1

        # Both items have EXACTLY the same probability
        # of being clicked.
        clicked = (
            rng.random()
            < CLICK_PROBABILITY
        )

        if clicked:

            click_counts[
                item
            ] += 1

            future_clicks[
                item
            ] += 1

    return {
        "seed_clicks_a":
            seed_clicks_a,

        "seed_clicks_b":
            seed_clicks_b,

        "future_clicks":
            future_clicks,

        "future_exposures":
            future_exposures,

        "final_click_counts":
            click_counts,
    }


# ---------------------------------------------------------
# SUMMARISE MANY PARALLEL WORLDS
# ---------------------------------------------------------

def run_condition(
    recommender_name,
    advantage,
):

    win_scores = []

    exposure_shares_a = []

    click_shares_a = []

    future_click_differences = []

    for world in range(
        NUM_WORLDS
    ):

        world_seed = (
            BASE_RANDOM_SEED
            + world
        )

        result = simulate_world(
            recommender_name,
            advantage,
            world_seed,
        )

        future_clicks = result[
            "future_clicks"
        ]

        future_exposures = result[
            "future_exposures"
        ]

        clicks_a = future_clicks[
            ITEM_A
        ]

        clicks_b = future_clicks[
            ITEM_B
        ]

        # ---------------------------------------------
        # WIN PROBABILITY
        # ---------------------------------------------
        #
        # A win counts as 1.
        # A loss counts as 0.
        # A tie counts as 0.5.

        if clicks_a > clicks_b:

            win_scores.append(
                1.0
            )

        elif clicks_a < clicks_b:

            win_scores.append(
                0.0
            )

        else:

            win_scores.append(
                0.5
            )

        # ---------------------------------------------
        # FUTURE EXPOSURE SHARE
        # ---------------------------------------------

        exposure_share_a = (
            future_exposures[
                ITEM_A
            ]
            / future_exposures.sum()
        )

        exposure_shares_a.append(
            exposure_share_a
        )

        # ---------------------------------------------
        # FUTURE CLICK SHARE
        # ---------------------------------------------

        total_future_clicks = (
            future_clicks.sum()
        )

        if total_future_clicks > 0:

            click_share_a = (
                clicks_a
                / total_future_clicks
            )

            click_shares_a.append(
                click_share_a
            )

        future_click_differences.append(
            clicks_a - clicks_b
        )

    return {
        "win_probability":
            np.mean(
                win_scores
            ),

        "mean_exposure_share_a":
            np.mean(
                exposure_shares_a
            ),

        "mean_click_share_a":
            np.mean(
                click_shares_a
            ),

        "mean_future_click_difference":
            np.mean(
                future_click_differences
            ),
    }


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    print(
        "=============================================="
    )

    print(
        "EARLY ADVANTAGE EXPERIMENT"
    )

    print(
        "=============================================="
    )

    print(
        f"Parallel worlds per condition: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Future interactions per world: "
        f"{FUTURE_INTERACTIONS:,}"
    )

    print(
        f"True click probability of both items: "
        f"{CLICK_PROBABILITY:.2f}"
    )

    print(
        f"Base seed clicks per item: "
        f"{BASE_SEED_CLICKS}"
    )

    print(
        f"Popularity alpha: "
        f"{POPULARITY_ALPHA}"
    )

    print()

    for recommender in [
        "random",
        "popularity",
    ]:

        print(
            f"--- {recommender.upper()} ---"
        )

        print(
            "Advantage | A win prob | "
            "A exposure | A clicks | "
            "Future click diff"
        )

        print(
            "-" * 67
        )

        for advantage in ADVANTAGE_LEVELS:

            metrics = run_condition(
                recommender,
                advantage,
            )

            print(
                f"{advantage:>8.0%} | "
                f"{metrics['win_probability']:>10.3f} | "
                f"{metrics['mean_exposure_share_a']:>10.3f} | "
                f"{metrics['mean_click_share_a']:>8.3f} | "
                f"{metrics['mean_future_click_difference']:>17.2f}"
            )

        print()

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()