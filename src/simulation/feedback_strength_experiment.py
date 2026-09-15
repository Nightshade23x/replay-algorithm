import numpy as np


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 500
FUTURE_INTERACTIONS = 5000

BASE_SEED_CLICKS = 100

ADVANTAGE_LEVELS = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.20,
]

ALPHA_LEVELS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]

CLICK_PROBABILITY = 0.50

BASE_RANDOM_SEED = 2026

ITEM_A = 0
ITEM_B = 1


# ---------------------------------------------------------
# RECOMMENDER
# ---------------------------------------------------------

def popularity_recommender(
    rng,
    click_counts,
    alpha,
):

    weights = (
        click_counts.astype(float)
        + 1.0
    ) ** alpha

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
    advantage,
    alpha,
    world_seed,
):

    rng = np.random.default_rng(
        world_seed
    )

    seed_clicks_a = round(
        BASE_SEED_CLICKS
        * (1 + advantage)
    )

    seed_clicks_b = (
        BASE_SEED_CLICKS
    )

    click_counts = np.array(
        [
            seed_clicks_a,
            seed_clicks_b,
        ],
        dtype=int,
    )

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

        item = popularity_recommender(
            rng,
            click_counts,
            alpha,
        )

        future_exposures[
            item
        ] += 1

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

    return (
        future_clicks,
        future_exposures,
    )


# ---------------------------------------------------------
# RUN ONE CONDITION
# ---------------------------------------------------------

def run_condition(
    advantage,
    alpha,
):

    win_scores = []
    exposure_shares = []
    click_shares = []

    for world in range(
        NUM_WORLDS
    ):

        world_seed = (
            BASE_RANDOM_SEED
            + world
        )

        (
            future_clicks,
            future_exposures,
        ) = simulate_world(
            advantage,
            alpha,
            world_seed,
        )

        clicks_a = future_clicks[
            ITEM_A
        ]

        clicks_b = future_clicks[
            ITEM_B
        ]

        # A wins = 1
        # tie = 0.5
        # loss = 0

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

        exposure_shares.append(
            future_exposures[
                ITEM_A
            ]
            / future_exposures.sum()
        )

        total_clicks = (
            future_clicks.sum()
        )

        click_shares.append(
            clicks_a
            / total_clicks
        )

    return {
        "win_probability":
            np.mean(
                win_scores
            ),

        "exposure_share":
            np.mean(
                exposure_shares
            ),

        "click_share":
            np.mean(
                click_shares
            ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "=========================================================="
    )

    print(
        "FEEDBACK STRENGTH EXPERIMENT"
    )

    print(
        "=========================================================="
    )

    print(
        f"Worlds per condition: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Interactions per world: "
        f"{FUTURE_INTERACTIONS:,}"
    )

    print()

    for alpha in ALPHA_LEVELS:

        print(
            f"--- ALPHA = {alpha:.1f} ---"
        )

        print(
            "Advantage | A win prob | "
            "A exposure | A clicks"
        )

        print(
            "-" * 51
        )

        for advantage in (
            ADVANTAGE_LEVELS
        ):

            metrics = run_condition(
                advantage,
                alpha,
            )

            print(
                f"{advantage:>8.0%} | "
                f"{metrics['win_probability']:>10.3f} | "
                f"{metrics['exposure_share']:>10.3f} | "
                f"{metrics['click_share']:>8.3f}"
            )

        print()

    print(
        "=========================================================="
    )


if __name__ == "__main__":
    main()